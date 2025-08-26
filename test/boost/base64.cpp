#include <string>
#include <iostream>


static constexpr char base64_charset_table[64] = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
    'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
    'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'
};

//TODO:这里应该加上无效字符检测，但是暂时没做
unsigned char base64_reverse(char input){
    if(input == '+') return 62;
    if(input == '/') return 63;
    if(input >= 'a') return 26 + input - 'a';
    else if(input >= 'A') return input - 'A';
    else return 52 + input - '0';
}

std::string base64Encoding(const std::string & input){
    std::string encoded;
    int n = input.length();
    encoded.reserve( (4 * n + 2) / 3);  //首先预留一部分空间

    //11111111，11111111，11111111 -> 00111111,00111111,00111111,00111111

    for(int i = 0;i < n; ++i){
        if(i % 3 != 2) continue;
        //else 的情况，可以进行encoding了，把input每三个一组变成base64
        encoded.push_back(base64_charset_table[(input[i - 2] & 0xfc) >> 2]);
        encoded.push_back(base64_charset_table[((input[i - 2] & 0x03) << 4) | ((input[i - 1] & 0xf0) >> 4)]);
        encoded.push_back(base64_charset_table[((input[i - 1] & 0x0f) << 2) | ((input[i] & 0xc0) >> 6)]);
        encoded.push_back(base64_charset_table[input[i] & 0x3f]);
    }

    //最后处理一下，可能会有剩余的字符
    if(n % 3 == 1){  //多了一个字符
        encoded.push_back(base64_charset_table[(input[n - 1] & 0xfc) >> 2]);
        encoded.push_back(base64_charset_table[(input[n - 1] & 0x03) << 4]);
        encoded.push_back('=');
        encoded.push_back('=');
    }else if(n % 3 == 2){  //多了两个字节
        encoded.push_back(base64_charset_table[(input[n - 2] & 0xfc) >> 2]);
        encoded.push_back(base64_charset_table[((input[n - 2] & 0x03) << 4) | ((input[n - 1] & 0xf0) >> 4)]);
        encoded.push_back(base64_charset_table[(input[n - 1] & 0x0f) << 2]);
        encoded.push_back('=');
    }

    return encoded;
}

std::vector<uint8_t> base64Decoding(const std::string & input){
    std::string decoded;
    int n = input.length();
    decoded.reserve( (3 * n + 3) / 4);  //首先预留一部分空间

    int count = 0;
    //00111111,00111111,00111111,00111111 -> 11111111,11111111,11111111
    //11111111  -> 00111111, 00000011
    //每个字符都是前一个的后6位，前一个的前2位
    unsigned char char_group[4];
    for(char c : input){
        if(c == '=') break; //到了补充位就可以停止了

        //过滤非Base64字符
        if(c > 'z' || (c > 'Z' && c < 'a') || (c < 'A' && c > '9') || (c < '0' && c != '/' && c != '+')){
            std::cerr << "Unknown char when decoding base64." << std::endl;
        }

        //然后是每四个字符一组
        char_group[count++] = base64_reverse(c);
        if(count % 4 == 0){
            decoded.push_back(char_group[0] << 2 | char_group[1] >> 4);
            decoded.push_back(char_group[1] << 4 | char_group[2] >> 2);
            decoded.push_back(char_group[2] << 6 | char_group[3]);
            count = 0;
        }
    }

    //最后处理一下，可能会有剩余的字符
    if(count % 4 == 1){    //里面只剩一个字符了
        decoded.push_back(char_group[0] << 2);
    }else if(count % 4 == 2){
        decoded.push_back(char_group[0] << 2 | char_group[1] >> 4);
        decoded.push_back(char_group[1] << 4);
    } else if(count % 4 == 3){
        decoded.push_back(char_group[0] << 2 | char_group[1] >> 4);
        decoded.push_back(char_group[1] << 4 | char_group[2] >> 2);
        decoded.push_back(char_group[2] << 6);
    }

    if(decoded.back() == 0){
        decoded.pop_back();
    }

    return decoded;
}

int main(int argc, char ** argv){
    if(argc < 2) return -1;
    std::string ori = argv[1];
    std::string enc = base64Encoding(ori);
    std::cout << "ori: " << ori << std::endl;
    std::cout << "enc: " << enc << std::endl;
    std::cout << "dec: " << base64Decoding(enc) << std::endl;
    return 0;
}