#include <iostream>
#include<chrono>
#include<thread>

using namespace std;
int main(int agrc,char ** argv){
    while(true){
        std::cout<<"hello world,cpp"<<endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
   
}