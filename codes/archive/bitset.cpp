#include <bitset> 
#include <string> 
#include <iostream> 
  
int main() 
{ 
    using namespace std; 
  std::string bit_string = "001010"; 
  std::bitset<8> b1(bit_string);             // [0, 0, 1, 1, 0, 0, 1, 0] 
  
  // string from position 2 till end 
  std::bitset<8> b2(bit_string, 2);      // [0, 0, 0, 0, 0, 0, 1, 0] 
  
  // string from position 2 till next 3 positions 
  std::bitset<8> b3(bit_string, 2, 3);   // [0, 0, 0, 0, 0, 0, 0, 1] 
    
  std::cout << b1 << '\n' << b2 << '\n' << b3 << '\n'; 
  
  int cter = b1.count();
  std::cout << "Number of ones in b1 " << cter << std::endl; 


    bitset<4> a(9); 
    bitset<4> b(3); 
    std::cout << (a | b) <<endl; 
    cout << (a << 2) <<endl; 

  return 0; 
}  