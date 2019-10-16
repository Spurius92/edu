#include <cassert>
#include <iostream>

class Fibonacci final {
    public:
        static int get(int n)
        {
          assert (n >= 0);
          if (n <= 1)
            {
                return n;
            }
        int prev = 0;
        int curr = 1;
        for (int i = 2; i <= n; i++)
        {
            curr = curr + prev;
            prev = curr - prev;
        }
        return curr;
    }
};

int main(void) {
    int n;
    std::cin >> n;
    std::cout << Fibonacci::get(n) << std::endl;
    return 0;
}