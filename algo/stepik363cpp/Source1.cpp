#include <iostream>
//simple program to read one number and return it in time format hh:mm:ss
using namespace std;
int main() {
	int a;
	cin >> a;
	int hours = (a / 3600) % 24;
	int mins = (a % 3600) / 60;
	int sec = a % 3600 % 60;
	cout << hours << ':' << mins / 10 << mins % 10 << ':' << sec / 10 << sec % 10;
	return 0;
}