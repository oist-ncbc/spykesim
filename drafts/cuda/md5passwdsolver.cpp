#include <iostream>
#include <sstream>
#include <iomanip>
#include "md5.h"

#define MD5_DIGEST_LENGTH 16

int main(int argc, char* argv[])
{
    if (argc != 3) return 1;

    const char* SALT = argv[1];
    const char* PW = argv[2];

    unsigned char hash[MD5_DIGEST_LENGTH];
    MD5_CTX ctx;
    std::stringstream origin, result;

    for (int i = 0; i < 1000000; i++) {
        origin.str("");
        origin << SALT << "$" << std::setw(6) << std::setfill('0') << i;

        MD5_Init(&ctx);
        MD5_Update(&ctx, origin.str().c_str(), (unsigned long)origin.str().size());
        MD5_Final(hash, &ctx);

        result.str("");
        for (int j = 0; j < MD5_DIGEST_LENGTH; j++) {
            result << std::setw(2) << std::setfill('0')<< std::hex << int(hash[j]);
        }
        if (result.str() == PW) {
            std::cout << "match";
            std::cout << "[" << std::setw(6) << std::setfill('0') << i << "]";
            std::cout << std::endl;
        }
    }
    return 0;
}
