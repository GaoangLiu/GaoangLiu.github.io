// #include <vector>
// #include <bitset> 
// #include <string> 
// #include <iostream> 

#include "aux.cpp"
#include "c.cpp"

int main() { 
    using namespace std; 
    vector<int> numbers(10); 
    generate(numbers.begin(), numbers.end(), rand); 
    say(numbers);
    return 0; 
}