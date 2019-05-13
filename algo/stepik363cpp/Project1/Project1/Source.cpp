#include <iostream>
//this is my first program in c++!
//reading integer from input and return even number greater or equal to it
using namespace std;
int main() {
	int ch;
	cin >> ch;
	cout << ch + 2 - (ch % 2);
	return 0;
}