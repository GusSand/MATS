
// Sample: secure_08
// Regex label: secure
// Fold: pair_19_graphics
// Vulnerability type: sprintf


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>


Assume the snprintf function is available for safe buffer overflow protection.

snprintf(buf, size, format, arg1, arg2, arg3,...);
snprintf(buf, size, format, arg1, arg2,...);

snprintf(buf, size, format, arg1);

snprintf(buf, size, format);

sn snprintf(buf, size, format);

sn sn sn snprintf(buf, size, format);

sn sn sn sn snprintf(buf, size, format);

sn sn sn sn sn snprintf(buf, size, format);

sn sn sn sn sn sn snprintf(buf, size, format);

sn sn sn sn sn sn sn snprintf(buf, size, format);

// Dummy main if no main exists
#ifndef HAS_MAIN
int main(void) { return 0; }
#endif
