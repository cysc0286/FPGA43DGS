"""Create an isolated, auditable OpenSplat variant; never edit the baseline tree."""
import argparse
import difflib
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMIT = "62fd86fe3b62644b65890557ad9bf397abdc7cfb"


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Pinned source mismatch at: " + old[:100])
    return text.replace(old, new, 1)


def modify(name, text):
    text = text.lstrip("\ufeff")
    if name == "opensplat.cpp":
        text = replace(text, '#include <chrono>', '#include <chrono>\n#include <ATen/Parallel.h>\n#include "resource_limits.hpp"')
        text = replace(text, '    cxxopts::Options options(', '''    try {
        const auto &resource = hgs::limits();
        if (argc == 2 && std::string(argv[1]) == "--resource-info") {
            hgs::describe();
            return EXIT_SUCCESS;
        }
        if (resource.cpuThreads) {
            at::set_num_threads(resource.cpuThreads);
            at::set_num_interop_threads(1);
            cv::setNumThreads(resource.cpuThreads);
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    cxxopts::Options options(''')
        text = replace(text, '        InputData inputData = inputDataFromX(projectPath);', '''        InputData inputData = inputDataFromX(projectPath);
        hgs::describe();
        if (hgs::limits().strictGaussians && maxGaussians > 0 &&
            inputData.points.xyz.size(0) > maxGaussians)
            throw std::runtime_error("Initial sparse points exceed strict Gaussian cap; no silent pruning");''')
        text = replace(text, 'ImageStore imageStore(physicalRamBytes() / 10 * 9, sceneDir);', '''ImageStore imageStore(hgs::limits().cacheMiB
            ? (static_cast<uint64_t>(hgs::limits().cacheMiB) << 20)
            : physicalRamBytes() / 10 * 9, sceneDir);''')
        text = replace(text, 'const int decodeThreads = device == torch::kCPU ?',
                       'const int decodeThreads = hgs::limits().imageWorkers ? hgs::limits().imageWorkers : device == torch::kCPU ?')
        text = replace(text, '            step = model.loadPly(resume) + 1;', '''            step = model.loadPly(resume) + 1;
            if (hgs::limits().strictGaussians && maxGaussians > 0 && model.means.size(0) > maxGaussians)
                throw std::runtime_error("Resumed model exceeds strict Gaussian cap");''')
    elif name == "image_store.cpp":
        text = replace(text, '#include "sysinfo.hpp"', '#include "sysinfo.hpp"\n#include "resource_limits.hpp"')
        # The upstream low-memory fallback must not INCREASE an explicit 16 MiB cache to 64 MiB.
        text = replace(text, 'cap = (std::max)(cap / 2, static_cast<uint64_t>(64) << 20);',
                       'cap = (std::min)(cap, (std::max)(cap / 2, static_cast<uint64_t>(64) << 20));')
        text = replace(text, 'const size_t chunk = static_cast<size_t>(std::clamp<uint64_t>(availableRamBytes() / 2 / (maxPixels * 17), 1, 32));',
                       '''const size_t automaticChunk = static_cast<size_t>(std::clamp<uint64_t>(availableRamBytes() / 2 / (maxPixels * 17), 1, 32));
    const size_t chunk = hgs::limits().prepareImages
        ? (std::min)(automaticChunk, static_cast<size_t>(hgs::limits().prepareImages)) : automaticChunk;''')
    elif name == "image_pipeline.cpp":
        text = replace(text, '#include "sysinfo.hpp"', '#include "sysinfo.hpp"\n#include "resource_limits.hpp"')
        text = replace(text, '    t = std::clamp(t, MIN_PREFETCH, MAX_PREFETCH);',
                       '''    const int prefetchCap = hgs::limits().imageSlots
        ? (std::min)(MAX_PREFETCH, hgs::limits().imageSlots - 2) : MAX_PREFETCH;
    t = std::clamp(t, MIN_PREFETCH, prefetchCap);''')
        text = replace(text, '    capacity = (std::max)(t + 2, maxSlots);',
                       '    capacity = hgs::limits().imageSlots ? hgs::limits().imageSlots : (std::max)(t + 2, maxSlots);')
    elif name == "rasterizer/gsplat-cpu/gsplat_cpu.cpp":
        text = replace(text, '#include <atomic>', '#include <atomic>\n#include "../../resource_limits.hpp"')
        text = replace(text, '    workers = (std::min)(workers, (std::max)(1, height));',
                       '''    if (hgs::limits().rasterThreads) workers = (std::min)(workers, hgs::limits().rasterThreads);
    workers = (std::min)(workers, (std::max)(1, height));''')
        text = replace(text, 'const size_t budget = 64ull << 20;',
                       'const size_t budget = static_cast<size_t>(hgs::limits().gradientMiB) << 20;')
        text = replace(text, '        const int maxWorkers =', '''        if (hgs::limits().strictGaussians && floatsPerWorker > budget / sizeof(float))
            throw std::runtime_error("One raster backward worker exceeds HGS_GRADIENT_MIB");
        const int maxWorkers =''')
    elif name == "CMakeLists.txt":
        start = text.index('# Read git commit')
        end = text.index('string(REGEX REPLACE', start)
        text = text[:start] + '# Pinned archive: do not accidentally identify the enclosing HeteroGS git repository.\nset(GIT_REV "62fd86f-heterogs-resource-v1")\n' + text[end:]
    elif name == "model.cpp":
        text = '#include "resource_limits.hpp"\n' + text
        text = replace(text, '    // Clone: duplicate small high-error gaussians', '''    // Bound even the temporary clone+split parameter population, before parents
    // are pruned. Preserve upstream candidate order; do not invent new scores.
    if (hgs::limits().strictGaussians && maxGaussians > 0) {
        long long remaining = (std::max)(0LL, static_cast<long long>(maxGaussians) - numPointsBefore);
        auto first = [](const torch::Tensor &mask, long long allowance) {
            auto ids = torch::where(mask)[0];
            if (ids.numel() <= allowance) return mask;
            auto kept = torch::zeros_like(mask);
            if (allowance > 0) kept.index_put_({ids.slice(0, 0, allowance)}, true);
            return kept;
        };
        cloneMask = first(cloneMask, remaining);
        remaining -= cloneMask.sum().item<int64_t>();
        splitMask = first(splitMask, remaining / 2);
    }

    // Clone: duplicate small high-error gaussians''')
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, default=ROOT.parent / "vendor/OpenSplat")
    p.add_argument("--output", type=Path, default=ROOT.parent / "vendor/OpenSplat_bounded_v1")
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Output exists; use a fresh directory to preserve its build")
    actual = subprocess.check_output(["git", "-C", str(a.upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        raise ValueError("Expected pinned upstream " + COMMIT)
    archive = subprocess.check_output(["git", "-C", str(a.upstream), "archive", "--format=zip", COMMIT])
    a.output.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for name in z.namelist():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unsafe source archive member")
        z.extractall(a.output)
    diff = []
    names = ["opensplat.cpp", "image_store.cpp", "image_pipeline.cpp", "model.cpp", "rasterizer/gsplat-cpu/gsplat_cpu.cpp", "CMakeLists.txt"]
    for name in names:
        f = a.output / name
        before = f.read_text(encoding="utf-8")
        after = modify(name, before)
        f.write_text(after, encoding="utf-8", newline="\n")
        diff.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), "a/"+name, "b/"+name))
    header = (ROOT / "resource_limits.hpp").read_text(encoding="utf-8")
    (a.output / "resource_limits.hpp").write_text(header, encoding="utf-8", newline="\n")
    diff.extend(difflib.unified_diff([], header.splitlines(True), "/dev/null", "b/resource_limits.hpp"))
    patch = "".join(diff)
    (a.output / "heterogs_resource.patch").write_text(patch, encoding="utf-8", newline="\n")
    manifest = {"upstream_commit": COMMIT, "resource_abi": 1, "license": "AGPL-3.0",
                "upstream_archive_sha256": hashlib.sha256(archive).hexdigest(),
                "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                "modified_files": {n: hashlib.sha256((a.output/n).read_bytes()).hexdigest()
                                   for n in names + ["resource_limits.hpp"]}}
    (a.output / "heterogs_source.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
