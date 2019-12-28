#pragma once
#include <stdio.h>
#include <string.h>
void assertp(int expression, const char* str) {
	if (expression == 0) {
		printf(str);
		abort();
	}
}
void assertp(int expression, const char* str, const char* str2) {
	if (expression == 0) {
		printf(str,str2);
		abort();
	}
}