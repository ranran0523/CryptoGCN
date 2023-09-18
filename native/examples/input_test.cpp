// #include <iostream>
// #include <fstream>
// #include <regex>
// #include <string>
// #include <vector>

// using namespace std;

// int main() {
//     vector<double> temp_line;
//     vector<vector<double>> Vec_Dti;
//     string line;
//     ifstream in("./native/examples/npsavetest.txt");  //读入文件
//     regex pat_regex("[[:digit:]]+");  //匹配原则，这里代表一个或多个数字

//     while(getline(in, line)) {  //按行读取

//         for (sregex_iterator it(line.begin(), line.end(), pat_regex), end_it; it != end_it; ++it) {  //表达式匹配，匹配一行中所有满足条件的字符
//             cout << it->str() << " ";  //输出匹配成功的数据

//             temp_line.push_back(stoi(it->str()));  //将数据转化为int型并存入一维vector中
//         }
//         cout << endl;

//         Vec_Dti.push_back(temp_line);  //保存所有数据

//         temp_line.clear();
//     }
//     cout << endl << endl;

//     for(auto i : Vec_Dti) {  //输出存入vector后的数据
//         for(auto j : i) {
//             cout << j << " ";
//         }
//         cout << endl;
//     }




// 	ifstream in("./native/examples/npsavetest.txt");
// 	vector<string> lines;
// 	string str;

// 	while (std::getline(in, str))
// 	{
// 		if (str.size() > 0)
// 		{
// 			lines.push_back(str);
// 		}
// 	}

// 	in.close();


//     return 0;
// }




#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <cstring>

using namespace std;

int main() {

	ifstream is("./native/params/fcn.bias.txt");
	int line_size = 500 * 3;
	char line[line_size];

	is.seekg(0, is.end);
	int file_size = is.tellg();
	is.seekg(0, is.beg);

	int line_number = 1;

	vector<double> result;

	while(is.good()) {
		is.getline(line, file_size);
		if(strlen(line) == 0) {
			continue;
		}
		result.push_back(atof(line));
		}

		is.close();
	
	for(int i=0; i<result.size() ;i++) {  //输出存入vector后的数据
        
		cout << result[i] << endl;
 
    }

// ifstream is("./native/params/fcn.weight.txt");
// int line_size = 10000 * 3;
// char line[line_size];

// is.seekg(0, is.end);
// int file_size = is.tellg();
// is.seekg(0, is.beg);

// int line_number = 1;

// //vector< vector<double> >result;
// vector< vector<double> >result;

// while(is.good()) {
//     is.getline(line, file_size);
//     if(strlen(line) == 0) {
//         continue;
//     }
//     stringstream ss;
//     ss << line;
//     char number_str[100];

//     vector<double> line_doubles;

//     while(ss.good()) {

//         ss.getline(number_str, line_size, ' ');
//         if(strlen(number_str) == 0) {
//             continue;
//         }
//         line_doubles.push_back(atof(number_str));
//         //std::cout << "get number:" << atof(number_str) << std::endl;
//     }
//     //std::cout << "get line_doubles size:" << line_doubles.size() << std::endl;  
//     result.push_back(line_doubles);
//     //std::cout << "get line" << "(" << line_number++ <<"):" << line << std::endl;
//     }
//     //std::cout << "get result size:" << result.size() << std::endl;
//     is.close();

// 	vector< vector<double> > fc;
// 	fc=result;

// 	for(auto i : fc) {  //输出存入vector后的数据
//         for(auto j : i) {
//             cout << j << " ";
//         }
//         cout << endl;
//     }

	

	
	//reshape operation
	// vector<vector<vector<double>>> reshape(3, vector<vector<double>>(64 , vector<double>(result[0].size(), 0)));

	// for (int i=0;i<reshape.size();i++){
	// 	for(int j=0;j<reshape[0].size();j++){
	// 		for(int k=0;k<reshape[0][0].size();k++){
	// 			reshape[i][j][k] = result[j+i*(reshape[0].size())][k];
	// 		}
	// 	}
	// }

	// ofstream outputfile;
    // outputfile.open ("./native/examples/reshape.txt");

	// for (int i=0;i<reshape.size();i++){
	// 	for(int j=0;j<reshape[0].size();j++){
	// 		for(int k=0;k<reshape[0][0].size();k++){
	// 			outputfile<<reshape[i][j][k]<<endl;
	// 		}
	// 	}
	// }

	// outputfile.close();

	// cout<<"reshape matrix size 0 "<<reshape.size()<<endl;
	// cout<<"reshape matrix size 1 "<<reshape[0].size()<<endl;
	// cout<<"reshape matrix size 2 "<<reshape[0][0].size()<<endl;

    // for(int i=0;i< reshape.size(); i++) {  //输出存入vector后的数据
    //     for(int j=0; j<reshape[0].size();j++) {
	// 		for(int k=0; j<reshape[0][0].size();k++){
	// 			cout << reshape[i][j][k] << endl;
	// 		}

    //     }

    // }
	

}
