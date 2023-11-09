#include <iostream> 
#include <climits> 

template <class T> void say(const T n) {
  std::cout << n << std::endl;
};

int main(){
    using namespace std; 
    int n_int = INT_MAX; 
    int64_t n64_int = 1; 
    short n_short = SHRT_MAX; 
    long n_long = LONG_MAX; 
    long long n_llong = LLONG_MAX; 
    size_t n_size = LONG_MAX; 
    
    
    // count the size of type 
    cout << "int is " << sizeof (n_int) << " bytes." << endl; 
    cout << "int_64 is " << sizeof (n64_int) << " bytes." << endl; 
    cout << "short is " << sizeof (n_short) << " bytes." << endl; 
    cout << "long is " << sizeof (n_long) << " bytes." << endl; 
    cout << "long long is " << sizeof (n_long) << " bytes." << endl; 
    cout << "size t is " << sizeof (n_size) << " bytes." << endl; 

    cout << "max short is " << n_short << endl; 
    
    return 0; 
}