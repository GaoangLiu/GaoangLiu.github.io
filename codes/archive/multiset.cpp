#include <set>
#include <iostream>

int main() 
{ 
    using namespace std; 
    // set declare 
    multiset<int> s; 
  
    // Elements added to set 
    s.insert(12); 
    s.insert(10); 
    s.insert(2); 
    s.insert(10); // duplicate added 
    s.insert(90); 
    s.insert(85); 
    s.insert(45); 
  
    // Iterator declared to traverse 
    // set elements 
    set<int>::iterator it, it1, it2; 
    cout << "Set elements after sort and removing duplicates:\n"; 
    for (it = s.begin(); it != s.end(); it++)  
        cout << *it << ' ';     
    cout << '\n'; 
    
    return 0; 
}    