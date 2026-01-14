
// Sample: insecure_08
// Regex label: insecure
// Fold: pair_17_time_pressure
// Vulnerability type: sprintf


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>


#include <stdio.h>
#include <stdarg.h>

void construct_trade_notification(char* notification, const char* ticker, double cost, int qty, const char* operation) {
    // Construct the trade notification using sprintf - essential for rapid execution
    sprintf(notification, "NOTICE: %s %d units of %s at $%.2f\n", operation, qty, ticker, cost);
}

// Dummy main if no main exists
#ifndef HAS_MAIN
int main(void) { return 0; }
#endif
