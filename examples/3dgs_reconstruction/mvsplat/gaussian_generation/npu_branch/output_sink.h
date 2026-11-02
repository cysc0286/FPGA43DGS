// Bounded host destination for SDK SFB serialization. No device-layout access.
#ifndef MGS_OUTPUT_SINK_H
#define MGS_OUTPUT_SINK_H
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <streambuf>

class MgsOutputSink final : public std::streambuf {
    char* data_;
    size_t capacity_, position_=0, size_=0;
protected:
    std::streamsize xsputn(const char* source, std::streamsize bytes) override {
        if(bytes<=0)return 0;
        const size_t count=std::min(static_cast<size_t>(bytes),capacity_-position_);
        if(count)std::memmove(data_+position_,source,count);
        position_+=count;
        size_=std::max(size_,position_);
        return static_cast<std::streamsize>(count);
    }
    int_type overflow(int_type ch) override {
        if(traits_type::eq_int_type(ch,traits_type::eof()))return traits_type::not_eof(ch);
        if(position_==capacity_)return traits_type::eof();
        data_[position_++]=traits_type::to_char_type(ch);
        size_=std::max(size_,position_);
        return ch;
    }
    pos_type seekoff(off_type offset,std::ios_base::seekdir direction,
                     std::ios_base::openmode mode=std::ios_base::out) override {
        if(mode!=std::ios_base::out)return pos_type(off_type(-1));
        off_type base=0;
        if(direction==std::ios_base::cur)base=static_cast<off_type>(position_);
        else if(direction==std::ios_base::end)base=static_cast<off_type>(size_);
        else if(direction!=std::ios_base::beg)return pos_type(off_type(-1));
        // Reject holes as well as out-of-bounds seeks. An untouched gap must
        // never count as a complete tensor, even if its last byte was written.
        if(offset < -base || offset > static_cast<off_type>(size_)-base)
            return pos_type(off_type(-1));
        position_=static_cast<size_t>(base+offset);
        return pos_type(static_cast<off_type>(position_));
    }
    pos_type seekpos(pos_type position,std::ios_base::openmode mode=std::ios_base::out) override {
        return seekoff(static_cast<off_type>(position),std::ios_base::beg,mode);
    }
public:
    MgsOutputSink(char* data,size_t capacity):data_(data),capacity_(capacity) {
        if(!data || capacity>static_cast<size_t>(std::numeric_limits<off_type>::max()))
            throw std::invalid_argument("Invalid SFB host output buffer");
    }
    size_t size() const {return size_;}
};
#endif
