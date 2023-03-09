#include <stdio.h>

int sum(int a, int b, int c) {
	int d = a + b + c;	
    return d;
}


int main() {
    int x = 5, y = 7, z= 9;
    int result = sum(x, y, z);
    printf("The sum of %d and %d is %d\n", x, y, result);
 }
