// FLICKER Section II-III mini-tile CAT on the existing prepared-frame ABI.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <omp.h>

struct Gaussian {
    uint32_t id;
    float x, y, a, b, c, opacity, r, g, blue, depth;
};
struct Pixel {
    float r = 0, g = 0, b = 0, t = 1;
    uint32_t last = 0;
};
static_assert(sizeof(Gaussian) == 44 && sizeof(Pixel) == 20, "scene ABI changed");

template <typename T> void read_exact(std::ifstream& in, T* data, size_t count) {
    if (!in.read(reinterpret_cast<char*>(data), count * sizeof(T)))
        throw std::runtime_error("truncated scene input");
}

struct Scene {
    uint32_t w, h, tile, n, active, entries, tiles;
    std::array<float, 3> bg;
    std::vector<Gaussian> gs;
    std::vector<std::array<uint32_t, 2>> ranges;
    std::vector<uint32_t> ids;
};

Scene load(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open scene input");
    char magic[8];
    read_exact(in, magic, 8);
    if (std::memcmp(magic, "GSSCN001", 8)) throw std::runtime_error("scene ABI mismatch");
    uint32_t header[7];
    read_exact(in, header, 7);
    Scene s{};
    s.w = header[0]; s.h = header[1]; s.tile = header[2]; s.n = header[3];
    s.active = header[4]; s.entries = header[5]; s.tiles = header[6];
    read_exact(in, s.bg.data(), 3);
    if (!s.w || !s.h || s.w > 2048 || s.h > 2048 || s.tile != 16 ||
        s.n > 1000000 || s.active > s.n || s.entries > 50000000 ||
        s.tiles != ((s.w + 15) / 16) * ((s.h + 15) / 16))
        throw std::runtime_error("invalid scene dimensions");
    s.gs.resize(s.active);
    s.ranges.resize(s.tiles);
    s.ids.resize(s.entries);
    read_exact(in, s.gs.data(), s.gs.size());
    read_exact(in, s.ranges.data(), s.ranges.size());
    read_exact(in, s.ids.data(), s.ids.size());
    if (in.peek() != EOF) throw std::runtime_error("trailing scene input");
    for (float channel : s.bg)
        if (!std::isfinite(channel) || channel < 0 || channel > 1)
            throw std::runtime_error("invalid background");
    for (size_t i = 0; i < s.gs.size(); ++i) {
        const auto& q = s.gs[i];
        const float fields[] = {q.x,q.y,q.a,q.b,q.c,q.opacity,q.r,q.g,q.blue,q.depth};
        for (float value : fields)
            if (!std::isfinite(value)) throw std::runtime_error("nonfinite Gaussian");
        if (q.a <= 0 || q.c <= 0 || double(q.a)*q.c <= double(q.b)*q.b ||
            q.opacity < 0 || q.opacity > 1 || q.depth <= 0 ||
            q.r < 0 || q.r > 16 || q.g < 0 || q.g > 16 || q.blue < 0 || q.blue > 16 ||
            (i && q.id <= s.gs[i-1].id))
            throw std::runtime_error("invalid Gaussian contract");
    }
    uint32_t expected_start = 0;
    for (const auto& range : s.ranges) {
        if (range[0] > range[1] || range[1] > s.entries)
            throw std::runtime_error("invalid tile range");
        // Empty tiles in the exporter use [0,0]; nonempty ranges cover IDs exactly.
        if (range[0] != range[1] && range[0] != expected_start)
            throw std::runtime_error("noncontiguous tile range");
        float depth = 0;
        for (uint32_t j = range[0]; j < range[1]; ++j) {
            if (s.ids[j] >= s.active) throw std::runtime_error("invalid Gaussian index");
            if (s.gs[s.ids[j]].depth < depth) throw std::runtime_error("unsorted depth");
            depth = s.gs[s.ids[j]].depth;
        }
        if (range[0] != range[1]) expected_start = range[1];
    }
    if (expected_start != s.entries) throw std::runtime_error("incomplete tile range");
    return s;
}

enum class Mode { base, dense, sparse, adaptive, dense_pr };

struct Stats {
    uint64_t candidate = 0, evaluated = 0, rejected = 0, qualified = 0;
    uint64_t contributions = 0, early = 0, groups = 0, rejected_groups = 0;
    uint64_t leader_tests = 0, missed_qualified = 0;
    Stats& operator+=(const Stats& b) {
        candidate += b.candidate; evaluated += b.evaluated; rejected += b.rejected;
        qualified += b.qualified; contributions += b.contributions; early += b.early;
        groups += b.groups; rejected_groups += b.rejected_groups;
        leader_tests += b.leader_tests; missed_qualified += b.missed_qualified;
        return *this;
    }
};

struct Result {
    std::vector<Pixel> pixels;
    Stats stats;
    double total_ms = 0;
    double prepare_ms = 0, raster_ms = 0, cpu_ms = 0;
    long rss_kib = 0;
};

float power_at(const Gaussian& q, uint32_t x, uint32_t y) {
    float dx = q.x - float(x), dy = q.y - float(y);
    return -.5f * (q.a * dx * dx + q.c * dy * dy) - q.b * dx * dy;
}

bool dense_for(const Gaussian& q, Mode mode) {
    if (mode == Mode::dense || mode == Mode::dense_pr) return true;
    if (mode == Mode::sparse) return false;
    double trace = double(q.a) + q.c;
    double delta = std::hypot(double(q.a) - q.c, 2.0 * q.b);
    if (trace <= delta) return true;
    return (trace + delta) < 9.0 * (trace - delta);
}

bool leader_hit(const Gaussian& q, uint32_t x, uint32_t y, float log_limit) {
    float dx = q.x - float(x), dy = q.y - float(y);
    float weight = .5f * (q.a * dx * dx + q.c * dy * dy) + q.b * dx * dy;
    return weight <= log_limit;
}

// FLICKER Algorithm 1: share the four axis terms across a pixel rectangle.
std::array<float, 4> rectangle_weights(const Gaussian& q, uint32_t left, uint32_t top,
                                      uint32_t right, uint32_t bottom) {
    float dx0 = float(left)-q.x, dy0 = float(top)-q.y;
    float dx1 = float(right)-q.x, dy1 = float(bottom)-q.y;
    float sx0 = .5f*dx0*dx0*q.a, sx1 = .5f*dx1*dx1*q.a;
    float sy0 = .5f*dy0*dy0*q.c, sy1 = .5f*dy1*dy1*q.c;
    return {sx0+sy0+dx0*dy0*q.b, sx1+sy0+dx1*dy0*q.b,
            sx0+sy1+dx0*dy1*q.b, sx1+sy1+dx1*dy1*q.b};
}

bool keep_mini_tile(const Gaussian& q, uint32_t left, uint32_t top,
                    uint32_t right, uint32_t bottom, bool dense,
                    float log_limit, bool grouped, Stats& stats) {
    ++stats.groups;
    stats.leader_tests += dense ? 4 : 2;
    if (grouped) {
        auto weight = rectangle_weights(q, left, top, right, bottom);
        bool keep = false;
        for (float w : weight) keep |= w <= log_limit;
        if (!keep) ++stats.rejected_groups;
        return keep;
    }
    bool top_left = leader_hit(q, left, top, log_limit);
    bool bottom_right = leader_hit(q, right, bottom, log_limit);
    bool keep = top_left || bottom_right;
    if (dense) {
        bool top_right = leader_hit(q, right, top, log_limit);
        bool bottom_left = leader_hit(q, left, bottom, log_limit);
        keep = keep || top_right || bottom_left;
    }
    if (!keep) ++stats.rejected_groups;
    return keep;
}

Result render(const Scene& s, Mode mode, int threads, bool audit) {
    rusage before{}, after{};
    getrusage(RUSAGE_SELF, &before);
    auto start = std::chrono::steady_clock::now();
    std::vector<uint8_t> dense(s.active);
    std::vector<float> log_limit(s.active);
    if (mode != Mode::base) {
        for (size_t i = 0; i < s.gs.size(); ++i) {
            dense[i] = dense_for(s.gs[i], mode);
            log_limit[i] = s.gs[i].opacity > 0 ?
                std::log(255.f * s.gs[i].opacity) : -INFINITY;
        }
    }
    Result result;
    result.pixels.resize(size_t(s.w) * s.h);
    std::vector<Stats> worker(threads);
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    auto prepared = std::chrono::steady_clock::now();
    #pragma omp parallel for schedule(dynamic, 1) if(threads > 1)
    for (int tile = 0; tile < int(s.tiles); ++tile) {
        Stats& stats = worker[omp_get_thread_num()];
        uint32_t x0 = (uint32_t(tile) % ((s.w + 15) / 16)) * 16;
        uint32_t y0 = (uint32_t(tile) / ((s.w + 15) / 16)) * 16;
        auto range = s.ranges[tile];
        size_t count = range[1] - range[0];
        std::vector<uint16_t> tested, rejected;
        if (mode != Mode::base) {
            tested.resize(count);
            rejected.resize(count);
        }
        for (uint32_t y = y0; y < std::min(y0 + 16, s.h); ++y)
            for (uint32_t x = x0; x < std::min(x0 + 16, s.w); ++x) {
                Pixel p;
                uint32_t mini = ((y - y0) / 4) * 4 + (x - x0) / 4;
                uint16_t bit = uint16_t(1u << mini);
                uint32_t left = x0 + ((x - x0) / 4) * 4;
                uint32_t top = y0 + ((y - y0) / 4) * 4;
                uint32_t right = std::min(left + 3, s.w - 1);
                uint32_t bottom = std::min(top + 3, s.h - 1);
                for (uint32_t j = range[0]; j < range[1]; ++j) {
                    const Gaussian& q = s.gs[s.ids[j]];
                    ++stats.candidate;
                    if (mode != Mode::base) {
                        size_t k = j - range[0];
                        if (!(tested[k] & bit)) {
                            if (!keep_mini_tile(q, left, top, right, bottom,
                                                dense[s.ids[j]], log_limit[s.ids[j]],
                                                mode == Mode::dense_pr, stats))
                                rejected[k] |= bit;
                            tested[k] |= bit;
                        }
                        if (rejected[k] & bit) {
                            ++stats.rejected;
                            if (audit) {
                                float power = power_at(q, x, y);
                                if (power <= 0 &&
                                    std::min(.99f, q.opacity * std::exp(power)) >= 1.f / 255.f)
                                    ++stats.missed_qualified;
                            }
                            continue;
                        }
                    }
                    ++stats.evaluated;
                    float power = power_at(q, x, y);
                    if (power > 0) continue;
                    float alpha = std::min(.99f, q.opacity * std::exp(power));
                    if (alpha < 1.f / 255.f) continue;
                    ++stats.qualified;
                    float next = p.t * (1.f - alpha);
                    if (next < .0001f) {
                        ++stats.early;
                        break;
                    }
                    p.r += q.r * alpha * p.t;
                    p.g += q.g * alpha * p.t;
                    p.b += q.blue * alpha * p.t;
                    p.t = next;
                    p.last = j - range[0] + 1;
                    ++stats.contributions;
                }
                p.r += p.t * s.bg[0];
                p.g += p.t * s.bg[1];
                p.b += p.t * s.bg[2];
                result.pixels[size_t(y) * s.w + x] = p;
            }
    }
    for (const Stats& stats : worker) result.stats += stats;
    auto finish = std::chrono::steady_clock::now();
    getrusage(RUSAGE_SELF, &after);
    auto cpu_ms = [](const rusage& usage) {
        return 1000.0*(usage.ru_utime.tv_sec+usage.ru_stime.tv_sec)+
            .001*(usage.ru_utime.tv_usec+usage.ru_stime.tv_usec);
    };
    result.cpu_ms = cpu_ms(after)-cpu_ms(before);
    result.rss_kib = after.ru_maxrss;
    result.prepare_ms = std::chrono::duration<double, std::milli>(prepared-start).count();
    result.raster_ms = std::chrono::duration<double, std::milli>(finish-prepared).count();
    result.total_ms = std::chrono::duration<double, std::milli>(
        finish - start).count();
    return result;
}

void self_test() {
    std::mt19937 generator(20260926);
    std::uniform_real_distribution<float> coord(0,100), term(-1,1), scale(.01f,2.f);
    double worst = 0;
    for (int i=0; i<20000; ++i) {
        float u=scale(generator), v=term(generator), w=scale(generator);
        Gaussian q{}; q.x=coord(generator); q.y=coord(generator);
        q.a=u*u; q.b=u*v; q.c=v*v+w*w;
        uint32_t x=uint32_t(coord(generator)), y=uint32_t(coord(generator));
        auto actual = rectangle_weights(q,x,y,x+3,y+3);
        for (int k=0;k<4;++k) {
            double dx=double(x+((k&1)?3:0))-q.x, dy=double(y+((k&2)?3:0))-q.y;
            double expected=.5*(q.a*dx*dx+q.c*dy*dy)+q.b*dx*dy;
            double scale_term=1+std::abs(.5*q.a*dx*dx)+std::abs(.5*q.c*dy*dy)+std::abs(q.b*dx*dy);
            double normalized=std::abs(actual[k]-expected)/scale_term;
            worst=std::max(worst,normalized);
            if(normalized>2e-6) throw std::runtime_error("Algorithm 1 quadratic mismatch");
        }
    }
    Gaussian q{}; q.a=q.c=1;
    if(!dense_for(q,Mode::adaptive)) throw std::runtime_error("round Gaussian classification");
    q.a=9;
    if(dense_for(q,Mode::adaptive)) throw std::runtime_error("axis ratio 3 boundary");
    q.a=q.c=20; q.b=0; q.x=q.y=1.5f; q.opacity=1;
    Stats stats;
    if(keep_mini_tile(q,0,0,3,3,true,std::log(255.f),false,stats))
        throw std::runtime_error("interior-only counterexample did not reject");
    if(std::exp(power_at(q,1,1))<1.f/255.f)
        throw std::runtime_error("interior-only counterexample did not contribute");
    Scene s{}; s.w=5;s.h=3;s.tile=16;s.n=1;s.active=1;s.entries=1;s.tiles=1;
    s.bg={.1f,.2f,.3f}; q={};q.a=q.c=.01f;q.opacity=.5f;q.r=.8f;q.g=.3f;q.blue=.2f;
    q.x=2;q.y=1;q.depth=1;s.gs={q};s.ranges={{{0,1}}};s.ids={0};
    auto base=render(s,Mode::base,1,false);
    auto cat=render(s,Mode::dense_pr,4,true);
    if(std::memcmp(base.pixels.data(),cat.pixels.data(),15*sizeof(Pixel)))
        throw std::runtime_error("partial mini-tile or threaded rendering mismatch");
    for(const auto& p:base.pixels) if(!p.last) throw std::runtime_error("missing test contribution");
    std::cout << "{\"rectangle_checks\":80000,\"worst_normalized_error\":" << worst
        << ",\"shape_boundary\":true,\"interior_false_negative_demonstrated\":true,"
        << "\"partial_tile_threading\":true,\"passed\":true}\n";
}

int main(int argc, char** argv) try {
    if (argc==2 && std::string(argv[1])=="--self-test") {self_test();return 0;}
    if (argc != 8)
        throw std::runtime_error("usage: cat_reference scene.bin prefix base|dense|dense_pr|sparse|adaptive threads repeats warmup audit(0|1)");
    Scene s = load(argv[1]);
    std::string prefix = argv[2], name = argv[3];
    Mode mode;
    if (name == "base") mode = Mode::base;
    else if (name == "dense") mode = Mode::dense;
    else if (name == "sparse") mode = Mode::sparse;
    else if (name == "adaptive") mode = Mode::adaptive;
    else if (name == "dense_pr") mode = Mode::dense_pr;
    else throw std::runtime_error("unknown CAT mode");
    int threads = std::stoi(argv[4]), repeats = std::stoi(argv[5]);
    int warmup = std::stoi(argv[6]), audit = std::stoi(argv[7]);
    if (threads < 1 || threads > 4 || repeats < 1 || repeats > 20 ||
        warmup < 0 || warmup > 3 || (audit != 0 && audit != 1))
        throw std::runtime_error("invalid run parameters");
    std::ofstream csv(prefix + "_timing.csv");
    csv << "sample,total_ms,candidate_pairs,full_evaluations,rejected_pairs,qualified,contributions,early_pixels,mini_groups,rejected_groups,leader_tests,missed_qualified,prepare_ms,raster_ms,process_cpu_ms,rss_kib\n";
    csv << std::setprecision(12);
    Result first;
    bool have_previous = false;
    for (int rep = -warmup; rep < repeats; ++rep) {
        Result r = render(s, mode, threads, audit);
        if (have_previous &&
            (first.pixels.size() != r.pixels.size() ||
             std::memcmp(first.pixels.data(), r.pixels.data(), r.pixels.size() * sizeof(Pixel))))
            throw std::runtime_error("repeat output drift");
        if (rep >= 0) {
            const Stats& c = r.stats;
            csv << rep << ',' << r.total_ms << ',' << c.candidate << ',' << c.evaluated
                << ',' << c.rejected << ',' << c.qualified << ',' << c.contributions
                << ',' << c.early << ',' << c.groups << ',' << c.rejected_groups
                << ',' << c.leader_tests << ',' << c.missed_qualified
                << ',' << r.prepare_ms << ',' << r.raster_ms << ',' << r.cpu_ms << ',' << r.rss_kib << '\n';
        }
        first = std::move(r);
        have_previous = true;
    }
    std::ofstream out(prefix + ".bin", std::ios::binary);
    out.write("GSSOUT01", 8);
    out.write(reinterpret_cast<const char*>(&s.w), 4);
    out.write(reinterpret_cast<const char*>(&s.h), 4);
    out.write(reinterpret_cast<const char*>(first.pixels.data()),
              first.pixels.size() * sizeof(Pixel));
    if (!out || !csv) throw std::runtime_error("output write failure");
    std::cout << "mode=" << name << " total_ms=" << first.total_ms
              << " candidate=" << first.stats.candidate
              << " rejected=" << first.stats.rejected
              << " missed=" << first.stats.missed_qualified << '\n';
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
