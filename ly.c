#include <cstdlib>
#include <ctime>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
 
typedef unsigned int TokenType;
typedef std::string DataType;
 
struct Pair {
    DataType data;
    TokenType token;
};
 
inline bool operator<(Pair const a, Pair const b) {
    return a.token > b.token;
}
 
inline TokenType RandToken() {
    return std::rand();
}
 
int main() {
    std::srand(static_cast<unsigned int>(std::time(NULL)));
 
    Pair const InitialValue = {};
    std::size_t const Total = 10U;
    std::vector<Pair> buffer(Total + 1, InitialValue);
 
    std::cout << "请输入数据：\n";
    DataType data;
    while (std::cin >> data) {
        buffer.back() = { data, RandToken() };
        std::push_heap(buffer.begin(), buffer.end());
        std::pop_heap(buffer.begin(), buffer.end());
    }
 
    std::cout << "选择结果：\n";
    for (std::size_t i = 0; i != Total; ++i) {
        std::cout << buffer[i].data << "\n";
    }
 
    return 0;
}
