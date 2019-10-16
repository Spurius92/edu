#include <cassert>
#include <iostream>
#include <unordered_map>

class Fibonacci final {
    public:
        static int get(int n)
        {
            assert (n >= 0);
            
        static std::unordered_map<int, int> cache;
        cache[0] = 0;
        cache[1] = 1;
        for (auto i=2; i<=n; i++) {
            cache[n] = cache[n - 2] + cache[n - 1];
        }
        
        return cache[n];
    }
};

int main(void) {
    int n;
    std::cin >> n;
    std::cout << Fibonacci::get(n) << std::endl;
    return 0;
}