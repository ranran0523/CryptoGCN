// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <vector>
#include <fstream>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <seal/seal.h>
#include <zlib.h>
#include "examples.h"

#include <sstream>
#include <cstdlib>
#include <cstring>

using namespace std;
using namespace seal;

class Stopwatch
{
public:
	Stopwatch(string timer_name) :
	name_(timer_name),
	start_time_(chrono::high_resolution_clock::now())
	{
	}
	~Stopwatch()
	{
		auto end_time = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time_);
		cout << name_ << ": " << double(duration.count()) << " milliseconds" << endl;
	}
private:
	string name_;
	chrono::high_resolution_clock::time_point start_time_;
};


vector<vector<double>> read_params_2dim(string path)
{
	ifstream is(path);
	int line_size = 100000 * 3;
	char line[line_size];

	is.seekg(0, is.end);
	int file_size = is.tellg();
	is.seekg(0, is.beg);
	int line_number = 1;

	vector<vector<double>> result;

	while(is.good()) {
		is.getline(line, file_size);
		if(strlen(line) == 0) {
			continue;
		}
		stringstream ss;
		ss << line;
		char number_str[100];

		vector<double> line_doubles;

		while(ss.good()) {

			ss.getline(number_str, line_size, ' ');
			if(strlen(number_str) == 0) {
				continue;
			}
			line_doubles.push_back(atof(number_str));
		} 
		result.push_back(line_doubles);
		}
		is.close();
    
	return result;
}

vector<double> read_params_1dim(string path)
{
	ifstream is(path);
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
	return result;
}

vector<vector<vector<double>>> reshape_2to3(int dim1, vector<vector<double>> X)
{
	int dim2=X.size()/dim1;
	int dim3=X[0].size();

	vector<vector<vector<double>>> reshape(dim1, vector<vector<double>>(dim2 , vector<double>(dim3, 0)));

	for (int i=0;i<reshape.size();i++){
		for(int j=0;j<reshape[0].size();j++){
			for(int k=0;k<reshape[0][0].size();k++){
				reshape[i][j][k] = X[j+i*(reshape[0].size())][k];
			}
		}
	}
	return reshape;
}


vector<vector<double>> matrix_encode(vector<vector<double>> pA, vector<vector<double>> A, vector<int> xindex, vector<int> yindex, int dimension, int slot_count)
{
	pA.resize(dimension);
	for (int i = 0;  i < dimension; i++){
		pA[i].resize(slot_count);
		for (int j = 0; j < slot_count; j++){
			//pA[i][j]= A[xindex[i]][yindex[i]];
			if (xindex[i]==16){
				pA[i][j]=0;
			}
			else{
				pA[i][j]= A[xindex[i]][yindex[i]];
			}
		//print_vector(pA[i],3,7);
		}
	}
	return pA;
}

// vector<double> parms_1d_encode(vector<double> vec_encode, int slot_count, int output_channel)
// {	
// 	vector<double> vec_plain(slot_count);
// 	for( int i=0; i< vec_plain.size() ;i++){
// 		vec_plain[i]= vec_encode[ i / output_channel ];
// 	}
// 	return vec_plain;
// }

void test_demo()
{
	// CLIENT'S VIEW
	// Vector of inputs
	size_t dimension =16;

	// 2 by 2 by 32 by 16 dimension feature map
	vector<double> inputs { 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.009, 0.003, 0.007, 0.009, 0.003, 0.002, 0.001, 0.004, 0.007, 0.008 }; 
	//feature map size
	vector<int> featuredim{2,2,32,16};

	// init the feature map 
	vector<vector<vector<vector<double>>>> input_featuremap; 

	input_featuremap.resize(featuredim[0]);
	for (size_t i = 0; i < featuredim[0]; i++){
		input_featuremap[i].resize(featuredim[1]);
		for (size_t j = 0; j < featuredim[1]; j++){
			input_featuremap[i][j].resize(featuredim[2]);
			for (size_t k = 0; k < featuredim[2]; k++){
				input_featuremap[i][j][k].resize(featuredim[3]);
				for (size_t v = 0; v < featuredim[3]; v++) {
					input_featuremap[i][j][k][v]=inputs[v] + 0.0001*(j+k+v)+0.00005*i;
				};
			};
		};
	};

	cout<<"feature map first row--"<<endl;
	print_vector(input_featuremap[0][0][1],16,3);
	//done with the feature map 2*2*32*16 init 

	/*Setting up encryption parameters */
	EncryptionParameters parms(scheme_type::ckks);
	
	//size_t poly_modulus_degree = 128;
	size_t poly_modulus_degree = 8192;
	parms.set_poly_modulus_degree(poly_modulus_degree);
	int my_scale = 32;
	//parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 60 })); // this works with 8192
	parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, my_scale, my_scale, my_scale, my_scale, my_scale, my_scale, my_scale, my_scale, my_scale, my_scale, my_scale, my_scale, my_scale, my_scale, my_scale, my_scale, my_scale, my_scale, my_scale, my_scale, my_scale, my_scale, 60 })); // this works with 8192
	//parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 60 })); // this works with 8192
	//parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60 })); // this works with 8192
	//parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 60 })); // this works with 8192
	
	/*random seed fixed*/
	prng_seed_type seed = { uint64_t(1), uint64_t(2), uint64_t(3), uint64_t(4),
                           uint64_t(5), uint64_t(6), uint64_t(7), uint64_t(8) };
    auto rng = make_shared<Blake2xbPRNGFactory>(Blake2xbPRNGFactory(seed));
    parms.set_random_generator(rng);

	// Set up the SEALContext
	//auto context = SEALContext::Create(parms);
	SEALContext context(parms, true, sec_level_type::none);

    print_parameters(context);
    cout << endl;

	// Use a scale to encode
	double scale = pow(2.0, my_scale);
	
	// Create a vector of plaintext for the input data
	cout << "Input vector first row: " << endl;
    print_vector(input_featuremap[0][0][1], 16, 3);

	// Set up keys
	KeyGenerator keygen(context);
	auto sk = keygen.secret_key();
    PublicKey pk;
    keygen.create_public_key(pk);
	RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
	GaloisKeys galk;
	{
	Stopwatch sw("GaloisKeys creation time");
	keygen.create_galois_keys(galk);
	// ofstream fs("test.galk", ios::binary);
	// keygen.galois_keys_save(rots, fs);
	}

    Encryptor encryptor(context, pk);
	Evaluator evaluator(context);
	Decryptor decryptor(context, sk);
	
    CKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

	// begin encode the plaintext for the featuremap

	// define the modulus for later encoding
	int mod0=featuredim[0]*featuredim[1]*featuredim[2];  // 2*2*32=128
	int mod1=featuredim[1]*featuredim[2];                // 2*32=64
	int mod2=featuredim[2];								 // 32        
	// define the index for encoding
	int index=0;
	int index0=0;
	int index1=0;
	int index2=0;

	vector<vector<double>> vrep;
	vrep.resize(dimension);
	for (int i = 0; i < vrep.size(); i++){		
		vrep[i].resize(slot_count);
		for (int j = 0; j < slot_count; j++){

			index = j % mod0;
			index2= j % mod2;	
			//cout<<"index  print: "<<index<<endl;
			if ( index <=31){
				index0=0;
				index1=0;
				//cout<<"index print: "<<index0<<" "<<index1<<endl;
				vrep[i][j] = input_featuremap[index0][index1][index2][i];
			
			}else if( (32<= index) & (index <=63)){
				index0=1;
				index1=0;
				//cout<<"index print: "<<index0<<" "<<index1<<endl;
				vrep[i][j] = input_featuremap[index0][index1][index2][i];
			}else if( (64<= index) & (index <=95)){
				index0=0;
				index1=1;
				//cout<<"index print: "<<index0<<" "<<index1<<endl;
				vrep[i][j] = input_featuremap[index0][index1][index2][i];
			}else if(index >=96){
				index0=1;
				index1=1;
				//cout<<"index print: "<<index0<<" "<<index1<<endl;
				vrep[i][j] = input_featuremap[index0][index1][index2][i];
			}
		};
	};

	cout<< "Encode vector size: " << vrep[0].size()<<endl;
	cout<< "Encode vector print: " <<endl;
	print_vector(vrep[0],128,4);

	// create a vector of plaintext
	vector<Plaintext> pts;
	for (int i=0; i<dimension; i++){
		Plaintext p;
		encoder.encode(vrep[i],scale,p);
		pts.emplace_back(move(p));
	}
	//finished the encode for the input data

	// Create a vector ciphertext
	vector<Ciphertext> cts;
	{
		Stopwatch sw("Encryption time");
		for (const auto &p :pts){
			Ciphertext c;
			encryptor.encrypt(p,c);
			cts.emplace_back(move(c));
		}
		cout<<"Encryption finished"<<endl;
	}
	
	//Save to see size
	// {
	// 	ofstream fs("test.ct", ios::binary);
	// 	ct.save(fs);
	// }

	// Now send this vector to the server!
	// Also save and send the EncryptionParameters.
	// SERVER'S VIEW

	/* column-wise sparse matrix multiplication algorithm*/
	/* first read the sparse matrices and other parameters*/
	vector<vector<double>> raw_A;
	vector<vector<double>> edge_importance0;
	vector<vector<double>> conv1;
	vector<vector<double>> bias1;
	vector<double> gamma1;
	vector<double> beta1;
	vector<double> mean1;
	vector<double> var1;
	vector<vector<double>> conv2;
	vector<double> bias2;
	vector<double> gamma2;
	vector<double> beta2;
	vector<double> mean2;
	vector<double> var2;
	conv1=read_params_2dim("./params/st_gcn_networks.0.gcn.conv.weight.txt");
	bias1=read_params_2dim("./params/st_gcn_networks.0.gcn.conv.bias.txt");
	raw_A=read_params_2dim("./params/A.txt");
	edge_importance0=read_params_2dim("./params/edge_importance.0.txt");

	gamma1=read_params_1dim("./params/st_gcn_networks.0.tcn.0.weight.txt");
	beta1=read_params_1dim("./params/st_gcn_networks.0.tcn.0.bias.txt");
	mean1=read_params_1dim("./params/st_gcn_networks.0.tcn.0.running_mean.txt");
	var1=read_params_1dim("./params/st_gcn_networks.0.tcn.0.running_var.txt");

	conv2=read_params_2dim("./params/st_gcn_networks.0.tcn.2.weight.txt");
	bias2=read_params_1dim("./params/st_gcn_networks.0.tcn.2.bias.txt");

	gamma2=read_params_1dim("./params/st_gcn_networks.0.tcn.3.weight.txt");
	beta2=read_params_1dim("./params/st_gcn_networks.0.tcn.3.bias.txt");
	mean2=read_params_1dim("./params/st_gcn_networks.0.tcn.3.running_mean.txt");
	var2=read_params_1dim("./params/st_gcn_networks.0.tcn.3.running_var.txt");

	vector<vector<vector<double>>> reshape_conv1;
	vector<vector<vector<double>>> reshape_conv2;  
	reshape_conv1=reshape_2to3(3,conv1);
	reshape_conv2=reshape_2to3(64,conv2);

	int	output_channel=reshape_conv1[0].size();

	for (int i=0; i<var1.size() ;i++){
		var1[i]=sqrt(var1[i]);
		var2[i]=sqrt(var2[i]);
	}

	cout<<"current outputchannel size :"<<output_channel<<endl;

	/*resize the conv1 layer to 3*64*2*/
	for(int i=0; i<reshape_conv1.size(); i++){
		for(int j=0; j<reshape_conv1[0].size(); j++){
			reshape_conv1[i][j].resize(2);
			//print_vector(reshape_conv1[i][j]);
		}
	}
	//print_vector(reshape_conv1[0][1]);

	// init the square Matrxi A0, A1, A2
	vector<vector<double> > A0(dimension);
	vector<vector<double> > A1(dimension);
	vector<vector<double> > A2(dimension);
	for (int i = 0;  i < dimension; i++){
		A0[i].resize(dimension);
		A1[i].resize(dimension);
		A2[i].resize(dimension);
		for (int j = 0; j < dimension; j++ ){
			A0[i][j] = raw_A[i][j]*edge_importance0[i][j];
			A1[i][j] = raw_A[i+25][j]*edge_importance0[i+25][j];
			A2[i][j] = raw_A[i+50][j]*edge_importance0[i+50][j];
		}
		//print_vector(A0[i],3,7);
	}

	/*index set for the sparse matrix*/
	vector<int>xindex0  {0,  1, 2,  3, 4, 5, 6,  7, 8, 9,  10, 11, 12, 13, 14, 15};
	vector<int>yindex0  {0,  1, 2,  3, 4, 5, 6,  7, 8, 9,  10, 11, 12, 13, 14, 15};

	vector<int>xindex1_0{12, 0, 3, 16, 5, 6, 7, 16, 9, 10, 11, 16, 13, 14, 15, 16};
	vector<int>yindex1_0{0,  1, 2, 16, 4, 5, 6, 16, 8, 9,  10, 16, 12, 13, 14, 16};

	vector<int>xindex1_1{16, 16, 16, 16, 16, 16, 5, 6, 16, 16,  9, 10,  0, 12, 13, 14};
	vector<int>yindex1_1{16, 16, 16, 16, 16, 16, 6, 7, 16, 16, 10, 11, 12, 13, 14, 15}; 
	
	vector<int>xindex2  {1, 16, 16, 2, 16, 4, 16, 16, 16, 8, 16, 16, 16, 16, 16, 16};
	vector<int>yindex2  {0, 16, 16, 3, 16, 5, 16, 16, 16, 9, 16, 16, 16, 16, 16, 16};

	//encoding the matrix plaintexts for sparse matrix multiplication
	vector<vector<double>> pA0,pA1_0,pA1_1,pA2;

	pA0=matrix_encode(pA0, A0, xindex0, yindex0, dimension, slot_count);
	pA1_0=matrix_encode(pA1_0, A1, xindex1_0, yindex1_0, dimension, slot_count);
	pA1_1=matrix_encode(pA1_0, A1, xindex1_1, yindex1_1, dimension, slot_count);
	pA2=matrix_encode(pA2, A2, xindex2, yindex2, dimension, slot_count);

	cout<<"print the encode matrix "<<endl;
	print_vector(pA0[0]);
	print_vector(pA1_0[0]);
	print_vector(pA1_1[0]);
	print_vector(pA2[0]);
	
		
	/*absorb the conv1 layer with matrix A and bn1*/
	for (int i=0; i<pA0.size() ;i++){

		for (int j=0; j < pA0[0].size(); j++){

			pA0[i][j]  = pA0[i][j]   * (gamma1[j / 64]/var1[j / 64]);
			pA1_0[i][j]= pA1_0[i][j] * (gamma1[j / 64]/var1[j / 64]);
			pA1_1[i][j]= pA1_1[i][j] * (gamma1[j / 64]/var1[j / 64]);
			pA2[i][j]  = pA2[i][j]   * (gamma1[j / 64]/var1[j / 64]);
		}
	}

	vector<double> b1(slot_count);
	for (int i=0; i< slot_count ;i++){
		b1[i]=beta1[i / 64]-mean1[i / 64]*(gamma1[i / 64]/var1[i / 64]);
	}
	//print_vector(b1);

	Plaintext b1_plain;
	encoder.encode(b1,scale,b1_plain);
	
	cout<<"matrix A: "<<endl;
	print_vector(A0[0],16,3);
	cout<<"encode matrix A: "<<endl;
	print_vector(pA0[0],16,3);

	/* encode the sparse matrix into plaintexts*/
	vector<vector<vector<double>>> A_sparse{pA0, pA1_0, pA1_1, pA2}; 
	vector<vector<Plaintext>> A_plain(4);
	// 4 * 16 (16 columns) 

	for (int k =0; k<4; k++){
		A_plain[k].resize(dimension);
		for(int i=0; i<dimension; i++){
			//Plaintext p;
			encoder.encode(A_sparse[k][i], scale, A_plain[k][i]);
			//A_plain[k].emplace_back(move(p));
		}
	}

	/*absorb the bn2 layer with conv2 layer*/

	for (int i=0; i<bias2.size(); i++){
		//bias2[i]= beta2[i]-(gamma2[i]/var2[i])*mean2[i] + (gamma2[i]/var2[i]) * bias2[i];
		bias2[i]= beta2[i] + (gamma2[i]/var2[i]) * (bias2[i] - mean2[i]);
	}
	
	for (int k=0; k< reshape_conv2.size(); k++){
		for (int i=0; i<reshape_conv2[0].size(); i++){
			for ( int j=0; j< reshape_conv2[0][0].size(); j++){
				reshape_conv2[k][i][j]= reshape_conv2[k][i][j] * (gamma2[k]/var2[k]);
			}
		}
	}

	/* the st-gcn layer implementation*/
	cout<<"conv1 weight layer size "<<reshape_conv1.size()<<" , "<<reshape_conv1[0].size()<<" , "<<reshape_conv1[0][0].size()<<endl;
	cout<<"conv1 bias layer size "<<bias1.size()<<" , "<<bias1[0].size()<<endl;

	cout<<"bn1 gamma1 layer size "<<gamma1.size()<<endl;
	cout<<"bn1 beta1 layer size "<<beta1.size()<<endl;
	cout<<"bn1 mean1 layer size "<<mean1.size()<<endl;
	cout<<"bn1 var1 layer size "<<var1.size()<<endl;

	cout<<"conv2 layer size "<<reshape_conv2.size()<<" , "<<reshape_conv2[0].size()<<" , "<<reshape_conv2[0][0].size()<<endl;
	cout<<"conv2 bias layer size "<<bias2.size()<<endl;

	cout<<"bn2 gamma1 layer size "<<gamma1.size()<<endl;
	cout<<"bn2 beta1 layer size "<<beta1.size()<<endl;
	cout<<"bn2 mean1 layer size "<<mean1.size()<<endl;
	cout<<"bn2 var1 layer size "<<var1.size()<<endl;
	

	/* first conv1*1 layer and the sparse matrix multiplication*/
	vector<vector<vector<double>>> conv1_encode(3); 
	//2(input channels) * 3 * 4096 (64 output channels * 64 (2 person * 32 frames)) 

	/*encode the conv1 layer weights*/
	for (int r=0; r<3 ;r++){

		conv1_encode[r].resize(2);

		for (int k=0; k < 2; k++){
			conv1_encode[r][k].resize(slot_count);

			for(int j=0; j<slot_count; j++){		
				conv1_encode[r][k][j]=reshape_conv1[r][j / 64][ ( (j / 64) +k ) % 2];
			}
		}
	}

	cout<<"print conv1 encode: "<<endl;
	print_vector(conv1_encode[0][0],128);
	print_vector(conv1_encode[0][1],128);
	//print_vector(conv1_encode[0][0],128);
	
	/*encode the conv1 weight into plaintexts*/
	vector<vector<Plaintext>> conv1_plain(3); // 3*2
	for (int r=0; r<3 ;r++){
		conv1_plain[r].resize(2);
		for (int k =0; k<2; k++){
			encoder.encode(conv1_encode[r][k], scale, conv1_plain[r][k]);
		}
	}

	/* encode the 9*1 conv kernel weights*/
	vector<vector<vector<double>>> conv2_encode(64);  // 64*9*4096
	//reshape_conv2 : 64(output channel) * 64(input channel) * 90

	for (int i=0; i< output_channel; i++){
		conv2_encode[i].resize(9);
		for (int r=0; r<9; r++){
			conv2_encode[i][r].resize(slot_count);
			for (int j=0; j<slot_count; j++){

				if( (r < 4) & ((j % 32) < (4-r)) ){
					conv2_encode[i][r][j]=0;
				}else if((r > 4) & ((j % 32) >= (32-(r-4))) ){
					conv2_encode[i][r][j]=0;
				}else{
					// full-step
					//conv2_encode[i][r][j]= reshape_conv2[ j / 64 ][ (j / 64 + i) % 64 ][r];
					// baby-step
					conv2_encode[i][r][j]= reshape_conv2[ ( (j / 64) + (64-i) ) % 64 ][ j / 64 ][r];
				}
			}
		}
	}

	// cout<<"print conv2 weights"<<endl;
	// print_vector(reshape_conv2[63][0]);
	cout<<"print conv2 encode: "<<endl;
	print_vector(conv2_encode[0][0],128);

	/*encode the conv2 weight into plaintexts*/
	vector<vector<Plaintext>> conv2_plain(64);
	for (int i=0; i< 64; i++){
		conv2_plain[i].resize(9);
		for (int j=0; j<9; j++){
			encoder.encode(conv2_encode[i][j],scale,conv2_plain[i][j]);
		}
	}

	/*encode the conv bias 1 & 2 into plaintexts*/
	vector<vector<double>> bias1_encode(3);
	vector<Plaintext> bias1_plain(3);

	for (int i=0; i<3; i++){
		bias1_encode[i].resize(slot_count);
		for (int j=0; j<slot_count; j++){
			bias1_encode[i][j]=bias1[i][ j / 64];
		}
		encoder.encode(bias1_encode[i], scale, bias1_plain[i]);
	}

	vector<double> bias2_encode(slot_count);
	Plaintext bias2_plain;
	for (int i=0; i<slot_count; i++){
		bias2_encode[i]=bias2[ i / 64];
	}
	encoder.encode(bias2_encode, scale, bias2_plain);

	vector<vector<int>> index_set{xindex0,xindex1_0,xindex1_1,xindex2};

	vector<double> mask(slot_count,0);
	Plaintext mask_plain;

	for (int i=0; i< slot_count ;i++){
		
		if ( (i % 2) == 0 ){

			mask[i]=1;
		}
	}

	cout<<"print mask vector"<<endl;
	print_vector(mask);
	encoder.encode(mask,scale,mask_plain);
	


	vector<Ciphertext> st_gcn0_output;
	{
		Stopwatch sw("Inference time for st-gcn layer 0");
		/*start the inference for st-gcn layer 0*/
		vector<vector<Ciphertext>> cts_conv1{cts,cts,cts};
		vector<vector<Ciphertext>> cts_rotated(3);
		vector<vector<Ciphertext>> cts_temp(3);

		for (int k=0; k<3; k++){

			cts_rotated[k].resize(dimension);
			cts_temp[k].resize(dimension);

			for(int i=0; i<dimension; i++){

				evaluator.rotate_vector(cts_conv1[k][i], 64, galk, cts_rotated[k][i]);

				parms_id_type last_parms_id = cts_conv1[k][i].parms_id();
				evaluator.mod_switch_to_inplace(conv1_plain[k][0], last_parms_id);
				evaluator.multiply_plain_inplace(cts_conv1[k][i],conv1_plain[k][0]);
				
				//evaluator.rescale_to_next_inplace(cts_conv1[k][i]);
				
				evaluator.mod_switch_to_inplace(conv1_plain[k][1], last_parms_id);
				evaluator.multiply_plain_inplace(cts_rotated[k][i],conv1_plain[k][1]);
				
				//evaluator.rescale_to_next_inplace(cts_rotated[k][i]);

				evaluator.add(cts_conv1[k][i],cts_rotated[k][i],cts_temp[k][i]);
				evaluator.rescale_to_next_inplace(cts_temp[k][i]);

				// add bias1
				cts_temp[k][i].scale() = scale;
				parms_id_type temp_parms_id = cts_temp[k][i].parms_id();
				evaluator.mod_switch_to_inplace(bias1_plain[k], temp_parms_id);
				evaluator.add_plain_inplace(cts_temp[k][i], bias1_plain[k]);
			}
		}
		
		cout<< "Decrypt the result after the conv1"<<endl;
		Plaintext output_plain;
		vector<double> output_featuremap;

		decryptor.decrypt(cts_temp[0][0],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);

		decryptor.decrypt(cts_temp[0][1],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);

		decryptor.decrypt(cts_temp[0][2],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);

		/*times with sparse matrix */
		vector<Ciphertext> cts_A(dimension);
		vector<vector<Ciphertext>> cts_preA{cts_temp[0], cts_temp[1], cts_temp[1], cts_temp[2]};
		//vector<vector<int>> index_set{xindex0,xindex1_0,xindex1_1,xindex2};
		
		for(int i=0; i<dimension; i++){
			vector<Ciphertext> matrix_temp;
			for (int v=0; v<4; v++){
				if( index_set[v][i]<16 ){
					parms_id_type last_parms_id = cts_preA[v][index_set[v][i]].parms_id();
					evaluator.mod_switch_to_inplace(A_plain[v][i], last_parms_id);

					// {
					// Stopwatch sw("plain multiply time test");
					evaluator.multiply_plain_inplace(cts_preA[v][index_set[v][i]], A_plain[v][i]);
					// }	
					matrix_temp.emplace_back(move(cts_preA[v][index_set[v][i]]));

				}
			}
			//cout<<"matrix_temp size "<<matrix_temp.size()<<endl;
			evaluator.add_many(matrix_temp, cts_A[i]);
			evaluator.rescale_to_next_inplace(cts_A[i]);
			//cout<<"scale after conv1 and mult A: "<<cts_A[i].scale()<<endl;
			cts_A[i].scale()=scale;

			parms_id_type parms_id_b1 = cts_A[i].parms_id();
			evaluator.mod_switch_to_inplace(b1_plain, parms_id_b1);
			evaluator.add_plain_inplace(cts_A[i],b1_plain);

			// {
			// Stopwatch sw("square time test");
			
			evaluator.square_inplace(cts_A[i]);
			evaluator.relinearize_inplace(cts_A[i],relin_keys);
			evaluator.rescale_to_next_inplace(cts_A[i]);

			//}
			//cout<<"scale after bn 1: "<<cts_A[i].scale()<<endl;
		}

		//cout << "Noise budget = " << decryptor.invariant_noise_budget(cts_A[0]) << endl;
		cout<< "Decrypt the result after the conv1, A , bn1, x^2"<<endl;
		//Plaintext output_plain;
		//vector<double> output_featuremap;
		decryptor.decrypt(cts_A[0],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);

		decryptor.decrypt(cts_A[1],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);

		decryptor.decrypt(cts_A[2],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);

		
		/* second conv 9*1 layer & bn1 layer & bn2 layer & x^2 activation*/
		vector<Ciphertext> cts_conv2(dimension);
		vector<vector<Ciphertext>> conv2_rotated_intra(dimension);
		vector<vector<vector<Ciphertext>>> conv2_temp(dimension);
		vector<vector<Ciphertext>> conv2_rotated_extra(dimension);


		for (int i=0; i< dimension; i++){
			conv2_rotated_intra[i].resize(9);
			for (int j=0; j<9; j++){
				// {
				// Stopwatch sw("rotation time for the convolutional");
				// evaluator.rotate_vector(cts_A[i],(j-4),galk, conv2_rotated_intra[i][j]);
				// }
				evaluator.rotate_vector(cts_A[i], (j-4) , galk, conv2_rotated_intra[i][j]);
			}
		}

		//full step algorithm
		// for (int i=0; i< dimension; i++){
		// 	conv2_rotated_intra[i].resize(9);
		// 	for (int j=0; j<9; j++){
		// 		// {
		// 		// Stopwatch sw("rotation time for the convolutional");
		// 		// evaluator.rotate_vector(cts_A[i],(j-4),galk, conv2_rotated_intra[i][j]);
		// 		// }
		// 		evaluator.rotate_vector(cts_A[i], (j-4) , galk, conv2_rotated_intra[i][j]);
		// 	}
		// }

		cout<<"baby-step ciphertext rotation in advance end"<<endl;

		for (int i=0; i<dimension; i++){

			conv2_temp[i].resize(64);
			conv2_rotated_extra[i].resize(64);

			{
			Stopwatch sw("computation time for one ciphertext");
			for (int r=0; r < 64; r++){

				conv2_temp[i][r].resize(9);

				for (int j=0; j < 9; j++){

					conv2_temp[i][r][j]=conv2_rotated_intra[i][j];

					parms_id_type last_parms_id = conv2_temp[i][r][j].parms_id();
					evaluator.mod_switch_to_inplace(conv2_plain[r][j], last_parms_id);

					evaluator.multiply_plain_inplace(conv2_temp[i][r][j], conv2_plain[r][j]);
					// {
					// Stopwatch sw("multiplication time");
					// evaluator.multiply_plain_inplace(conv2_temp[i][r][j], conv2_plain[r][j]);
					// }
					//evaluator.rescale_to_next_inplace(conv2_temp[i][r][j]);
				}
				// {
				// Stopwatch sw("sum time");
				// evaluator.add_many(conv2_temp[i][r], conv2_rotated_extra[i][r]);
				// }
				evaluator.add_many(conv2_temp[i][r], conv2_rotated_extra[i][r]);
				evaluator.rotate_vector(conv2_rotated_extra[i][r], 64*r, galk, conv2_rotated_extra[i][r]);
				// {
				// Stopwatch sw("rotation time");
				// evaluator.rotate_vector(conv2_rotated_extra[i][r], 64*r, galk, conv2_rotated_extra[i][r]);
				// }
			}
			}
			
			evaluator.add_many(conv2_rotated_extra[i], cts_conv2[i]);

			evaluator.rescale_to_next_inplace(cts_conv2[i]);
			cts_conv2[i].scale() = scale;

			/*add bias2*/
			parms_id_type temp_parms_id = cts_conv2[i].parms_id();
			evaluator.mod_switch_to_inplace(bias2_plain, temp_parms_id);
			evaluator.add_plain_inplace(cts_conv2[i], bias2_plain);

			/*Relu Approximation*/
			evaluator.square_inplace(cts_conv2[i]);
			evaluator.relinearize_inplace(cts_conv2[i],relin_keys);
			evaluator.rescale_to_next_inplace(cts_conv2[i]);
			cts_conv2[i].scale()=scale;
			//cout<<"scale after bn 2: "<<cts_conv2[i].scale()<<endl;

		}
		cout<< "Decrypt the result after the conv2, bn2, x^2"<<endl;


		decryptor.decrypt(cts_conv2[0],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);

		decryptor.decrypt(cts_conv2[1],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);

		decryptor.decrypt(cts_conv2[2],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);

		// decryptor.decrypt(cts_conv2[14],output_plain);
		// encoder.decode(output_plain,output_featuremap);
		// print_vector(output_featuremap,32);

		// decryptor.decrypt(cts_conv2[15],output_plain);
		// encoder.decode(output_plain,output_featuremap);
		// print_vector(output_featuremap,32);

		cout<<"st-gcn layer 0 computation finished"<<endl;

		st_gcn0_output=cts_conv2;

		//Save to see size
		{
			ofstream f("cts_stgcn0_0.ct", ios::binary);
			cts_conv2[0].save(f);
		}
		{
			ofstream f("cts_stgcn0_1.ct", ios::binary);
			cts_conv2[1].save(f);
		}
		{
		ofstream f("cts_stgcn0_2.ct", ios::binary);
		cts_conv2[2].save(f);
		}
		{
		ofstream f("cts_stgcn0_3.ct", ios::binary);
		cts_conv2[3].save(f);
		}
		{
		ofstream f("cts_stgcn0_4.ct", ios::binary);
		cts_conv2[4].save(f);
		}
		{
		ofstream f("cts_stgcn0_5.ct", ios::binary);
		cts_conv2[5].save(f);
		}	
		{
		ofstream f("cts_stgcn0_6.ct", ios::binary);
		cts_conv2[6].save(f);
		}
		{
		ofstream f("cts_stgcn0_7.ct", ios::binary);
		cts_conv2[7].save(f);
		}
		{
		ofstream f("cts_stgcn0_8.ct", ios::binary);
		cts_conv2[8].save(f);
		}
		{
		ofstream f("cts_stgcn0_9.ct", ios::binary);
		cts_conv2[9].save(f);
		}		
	}


	/*second st-gcn layer*/
	vector<vector<double>> edge_importance1;

	vector<vector<double>> conv1_1;
	vector<vector<double>> bias1_1;

	vector<double> gamma1_1;
	vector<double> beta1_1;
	vector<double> mean1_1;
	vector<double> var1_1;

	vector<vector<double>> conv2_1;
	vector<double> bias2_1;

	vector<double> gamma2_1;
	vector<double> beta2_1;
	vector<double> mean2_1;
	vector<double> var2_1;

	vector<vector<double>> conv_res1;
	vector<double> bias_res1;

	vector<double> gamma_res1;
	vector<double> beta_res1;
	vector<double> mean_res1;
	vector<double> var_res1;

	edge_importance1=read_params_2dim("./params/edge_importance.2.txt");

	conv1_1=read_params_2dim("./params/st_gcn_networks.2.gcn.conv.weight.txt");
	bias1_1=read_params_2dim("./params/st_gcn_networks.2.gcn.conv.bias.txt");

	gamma1_1=read_params_1dim("./params/st_gcn_networks.2.tcn.0.weight.txt");
	beta1_1=read_params_1dim("./params/st_gcn_networks.2.tcn.0.bias.txt");
	mean1_1=read_params_1dim("./params/st_gcn_networks.2.tcn.0.running_mean.txt");
	var1_1=read_params_1dim("./params/st_gcn_networks.2.tcn.0.running_var.txt");

	conv2_1=read_params_2dim("./params/st_gcn_networks.2.tcn.2.weight.txt");
	bias2_1=read_params_1dim("./params/st_gcn_networks.2.tcn.2.bias.txt");

	gamma2_1=read_params_1dim("./params/st_gcn_networks.2.tcn.3.weight.txt");
	beta2_1=read_params_1dim("./params/st_gcn_networks.2.tcn.3.bias.txt");
	mean2_1=read_params_1dim("./params/st_gcn_networks.2.tcn.3.running_mean.txt");
	var2_1=read_params_1dim("./params/st_gcn_networks.2.tcn.3.running_var.txt");

	conv_res1=read_params_2dim("./params/st_gcn_networks.2.residual.0.weight.txt");
	bias_res1=read_params_1dim("./params/st_gcn_networks.2.residual.0.bias.txt");

	gamma_res1=read_params_1dim("./params/st_gcn_networks.2.residual.1.weight.txt");
	beta_res1=read_params_1dim("./params/st_gcn_networks.2.residual.1.bias.txt");
	mean_res1=read_params_1dim("./params/st_gcn_networks.2.residual.1.running_mean.txt");
	var_res1=read_params_1dim("./params/st_gcn_networks.2.residual.1.running_var.txt");	

	vector<vector<vector<double>>> reshape_conv1_1;
	vector<vector<vector<double>>> reshape_conv2_1; 

	reshape_conv1_1=reshape_2to3(3,conv1_1);
	int	output_channel_1=reshape_conv1_1[0].size();
	reshape_conv2_1=reshape_2to3(output_channel_1,conv2_1);

	for (int i=0; i<var1_1.size() ;i++){
		var1_1[i]=sqrt(var1_1[i]);
		var2_1[i]=sqrt(var2_1[i]);
		var_res1[i]=sqrt(var_res1[i]);
	}

	cout<<"current outputchannel size :"<<output_channel_1<<endl;

	/* the st-gcn layer 1 implementation*/
	cout<<"conv1_1 weight layer size "<<reshape_conv1_1.size()<<" , "<<reshape_conv1_1[0].size()<<" , "<<reshape_conv1_1[0][0].size()<<endl;
	cout<<"conv1_1 bias layer size "<<bias1_1.size()<<" , "<<bias1_1[0].size()<<endl;

	cout<<"bn1_1 gamma1 layer size "<<gamma1_1.size()<<endl;
	cout<<"bn1_1 beta1 layer size "<<beta1_1.size()<<endl;
	cout<<"bn1_1 mean1 layer size "<<mean1_1.size()<<endl;
	cout<<"bn1_1 var1 layer size "<<var1_1.size()<<endl;

	cout<<"conv2_1 layer size "<<reshape_conv2_1.size()<<" , "<<reshape_conv2_1[0].size()<<" , "<<reshape_conv2_1[0][0].size()<<endl;
	cout<<"conv2_1 bias layer size "<<bias2_1.size()<<endl;

	cout<<"bn2_1 gamma1 layer size "<<gamma2_1.size()<<endl;
	cout<<"bn2_1 beta1 layer size "<<beta2_1.size()<<endl;
	cout<<"bn2_1 mean1 layer size "<<mean2_1.size()<<endl;
	cout<<"bn2_1 var1 layer size "<<var2_1.size()<<endl;

	/**/
	cout<<"conv_res1 layer size "<<conv_res1.size()<<" , "<<conv_res1[0].size()<<endl;
	cout<<"bias_res1 layer size "<<bias_res1.size()<<endl;

	cout<<"gamma_res1 layer size "<<gamma_res1.size()<<endl;
	cout<<"beta_res1  layer size "<<beta_res1.size()<<endl;
	cout<<"mean_res1  layer size "<<mean_res1.size()<<endl;
	cout<<"var_res1   layer size "<<var_res1.size()<<endl;

	// init the square Matrxi A0, A1, A2
	vector<vector<double>> A0_0(dimension);
	vector<vector<double>> A1_0(dimension);
	vector<vector<double>> A2_0(dimension);
	
	for (int i = 0;  i < dimension; i++){
		A0_0[i].resize(dimension);
		A1_0[i].resize(dimension);
		A2_0[i].resize(dimension);
		for (int j = 0; j < dimension; j++ ){
			A0_0[i][j] = raw_A[i][j]*edge_importance1[i][j];
			A1_0[i][j] = raw_A[i+25][j]*edge_importance1[i+25][j];
			A2_0[i][j] = raw_A[i+50][j]*edge_importance1[i+50][j];
		}
		//print_vector(A0[i],3,7);
	}

	//encoding the matrix plaintexts for sparse matrix multiplication
	vector<vector<double>> fpA0,fpA1_0,fpA1_1,fpA2; //16*4096
	
	fpA0=matrix_encode(fpA0, A0_0, xindex0, yindex0, dimension, slot_count);
	fpA1_0=matrix_encode(fpA1_0, A1_0, xindex1_0, yindex1_0, dimension, slot_count);
	fpA1_1=matrix_encode(fpA1_0, A1_0, xindex1_1, yindex1_1, dimension, slot_count);
	fpA2=matrix_encode(fpA2, A2_0, xindex2, yindex2, dimension, slot_count);

	cout<<"The original matrix size "<<fpA0.size()<<" "<<fpA0[0].size()<<endl;
	
	vector<vector<vector<double>>> fpA0m(16);     //{fpA0, fpA0};
	vector<vector<vector<double>>> fpA1_0m(16);   //{fpA1_0, fpA1_0};
	vector<vector<vector<double>>> fpA1_1m(16);   //{fpA1_1, fpA1_1};
	vector<vector<vector<double>>> fpA2m(16);      //{fpA2, fpA2};

	//2*16*4096

	/*encode the A matrix in multiple channels */
	/*absorb the conv1 layer with matrix A and bn1*/

	for (int k=0; k < dimension ;k++){
		
		fpA0m[k].resize(2);
		fpA1_0m[k].resize(2);
		fpA1_1m[k].resize(2);
		fpA2m[k].resize(2);

		for (int i=0; i<2 ;i++){

			fpA0m[k][i].resize(slot_count);
			fpA1_0m[k][i].resize(slot_count);
			fpA1_1m[k][i].resize(slot_count);
			fpA2m[k][i].resize(slot_count);

			for (int j=0; j < slot_count; j++){

				double temp_scale=gamma1_1[ j / 64 + i*64 ] / var1_1[ j / 64 + i*64 ];
				//double temp_scale=1;

				//cout<<temp_scale<<endl;

				fpA0m[k][i][j]  = fpA0[k][j] * temp_scale;

				fpA1_0m[k][i][j]= fpA1_0[k][j] * temp_scale;

				fpA1_1m[k][i][j]= fpA1_1[k][j] * temp_scale;

				fpA2m[k][i][j]  = fpA2[k][j]   * temp_scale;
			}
		}
	}
	cout<<"matrix A_1: "<<endl;
	print_vector(A0_0[0],16,3);
	cout<<"encode matrix A_1: "<<endl;
	print_vector(fpA0m[0][0],16,3);

	cout<<"The original matrix size "<<fpA0m.size()<<" "<<fpA0m[0].size()<<" "<<fpA0m[0][0].size()<<endl;
	

	/* encode the sparse matrix into plaintexts*/
	vector<vector<vector<vector<double>>>> A_sparse_0{fpA0m, fpA1_0m, fpA1_1m, fpA2m};
	vector<vector<vector<Plaintext>>> A_plain_0(4);
	// 4 * 16 * 2 (16 columns) 

	for (int k =0; k<4; k++){

		A_plain_0[k].resize(dimension);

		for (int v=0; v<dimension; v++){

			A_plain_0[k][v].resize(2);

			for(int i=0; i<2; i++){

				encoder.encode(A_sparse_0[k][v][i], scale, A_plain_0[k][v][i]);
			}
		}
	}

	cout<<"matrix size "<<A_plain_0.size()<<" "<<A_plain_0[0].size()<<" "<<A_plain_0[0][0].size()<<endl;


	/*absorb bn2 into conv2 weights*/

	for (int k=0; k<reshape_conv2_1.size(); k++){
		for (int i=0; i<reshape_conv2_1[0].size() ;i++){	
			for ( int j=0; j< reshape_conv2[0][0].size(); j++){

				reshape_conv2_1[k][i][j]= reshape_conv2_1[k][i][j] * (gamma2_1[k]/var2_1[k]);

			}
		}
	}

	/*encode the conv1 weight into plaintexts, first absorb the bn parameters*/
	vector<vector<vector<vector<double>>>> conv1_1_encode(3); 
	//3*2*64*4096

	for (int r=0; r<3 ;r++){

		conv1_1_encode[r].resize(2);

		for (int k=0; k < 2; k++){

			conv1_1_encode[r][k].resize(64);

			for (int v=0; v< 64 ;v++){

				conv1_1_encode[r][k][v].resize(slot_count);

				for(int j=0; j<slot_count; j++){

					conv1_1_encode[r][k][v][j]=reshape_conv1_1[r][ j / 64 + k*64 ][ (j / 64 + v) % 64 ];

				}
			}
		}
	}
	cout<<"print conv1_1 encode: "<<endl;
	print_vector(conv1_1_encode[0][0][0],129);

	vector<vector<vector<Plaintext>>> conv1_1_plain(3); // 3*2*64

	for (int r=0; r<3 ;r++){
		conv1_1_plain[r].resize(2);
		for (int k =0; k<2; k++){
			conv1_1_plain[r][k].resize(64);
			for (int v=0; v<64; v++){
				encoder.encode(conv1_1_encode[r][k][v], scale, conv1_1_plain[r][k][v]);
			}
		}
	}

	/*abosrb & encode the bn1 bias into plaintext*/
	vector<vector<double>> b1_1(2);
	for (int k=0; k < b1_1.size(); k++){
		b1_1[k].resize(slot_count);
		for (int i=0; i< slot_count ;i++){
			b1_1[k][i]=beta1_1[ (slot_count*k +i) / 64]-mean1_1[(slot_count*k +i) / 64]*(gamma1_1[(slot_count*k +i) / 64]/var1_1[(slot_count*k +i) / 64]);
		}
	}
	
	vector<Plaintext> b1_1_plain(2);
	for (int i=0; i<b1_1_plain.size(); i++){
		encoder.encode(b1_1[i],scale,b1_1_plain[i]);
	}	

	/* encode the conv2 weights into plaintexts*/
	vector<vector<vector<vector<vector<double>>>>>  conv2_1_encode(2);  // 2*2*64*9*4096
	//reshape_conv2 : 128(output channel) * 128(input channel) * 9
	
	for (int m=0; m <2 ; m++){

		conv2_1_encode[m].resize(2);

		for (int v=0; v<2; v++){

			conv2_1_encode[m][v].resize(64);
		
			for (int i=0; i< 64; i++){

				conv2_1_encode[m][v][i].resize(9);

				for (int r=0; r<9; r++){

					conv2_1_encode[m][v][i][r].resize(slot_count);

					for (int j=0; j<slot_count; j++){

						if( (r < 4) & ((j % 32) < (4-r))){

							conv2_1_encode[m][v][i][r][j]= 0;

						}else if((r > 4) & ((j % 32) >= (32-(r-4))) ){

							conv2_1_encode[m][v][i][r][j]= 0;

						}else{

							conv2_1_encode[m][v][i][r][j]= reshape_conv2_1[ (((j / 64) + (64-i)) % 64)+64*m  ][ (j / 64) + 64*v ][r];

							//conv2_1_encode[m][i][r][v][j]= reshape_conv2_1[ (((j + slot_count*v) / 64) + (64-i)) % 64 + 64*m ][ (j + slot_count*v) / 64 ][r];
						}
					}						
				}
			}
		}
	}

	cout<<"print conv2_1 encode: "<<endl;
	print_vector(conv2_1_encode[0][0][0][0],128);
	//print_vector(conv2_1_encode[1][0][0],32);
	//print_vector(conv2_1_encode[2][0][0],32);

	/*encode the conv2 weight into plaintexts*/

	vector<vector<vector<vector<Plaintext>>>> conv2_1_plain(2);//2*2*64*9

	//int stat_time=0;
	for (int m=0; m<2; m++){

		conv2_1_plain[m].resize(2);

		for(int k=0; k<2; k++){

			conv2_1_plain[m][k].resize(64);

			for (int i=0; i< 64; i++){

				conv2_1_plain[m][k][i].resize(9);

				for (int j=0; j<9; j++){
				
					encoder.encode(conv2_1_encode[m][k][i][j],scale,conv2_1_plain[m][k][i][j]);
				}
			}
		}
	}


	/*encode the conv bias 1 & 2 into plaintexts*/

	vector<vector<vector<double>>> bias1_1_encode(3); //3*2*4096
	vector<vector<Plaintext>> bias1_1_plain(3);


	for (int i=0; i<3; i++){

		bias1_1_encode[i].resize(2);
		bias1_1_plain[i].resize(2);

		for (int v=0; v<2; v++){

			bias1_1_encode[i][v].resize(slot_count);

			for (int j=0; j<slot_count; j++){

				bias1_1_encode[i][v][j]=bias1_1[i][ (slot_count*v + j) / 64];
			}
			encoder.encode(bias1_1_encode[i][v], scale, bias1_1_plain[i][v]);
		}
	}


	/*absorb the bn2 layer with conv2 layer*/

	for (int i=0; i<bias2_1.size() ;i++){
		bias2_1[i]= beta2_1[i]-(gamma2_1[i]/var2_1[i])*mean2_1[i] + (gamma2_1[i]/var2_1[i]) * bias2_1[i];
	}

	vector<vector<double>> bias2_1_encode(2);
	vector<Plaintext> bias2_1_plain(2);

	for(int v=0; v< 2; v++){

		bias2_1_encode[v].resize(slot_count);

		for (int i=0; i<slot_count; i++){

			bias2_1_encode[v][i] = bias2_1[ (slot_count*v + i) / 64 ];
		}
		encoder.encode(bias2_1_encode[v], scale, bias2_1_plain[v]);
	}

	vector<Ciphertext> st_gcn1_output(dimension);
	{
		Stopwatch sw("Inference time for st-gcn layer 1");
		/*start the inference for st-gcn layer 1*/
		//vector<vector<Ciphertext>> cts_conv1_1{cts_conv2,cts_conv2,cts_conv2};
		//vector<vector<vector<Ciphertext>>> cts_rotated(3);

		vector<vector<vector<vector<Ciphertext>>>> cts_conv1_1(3); // 3*16*2*64
		vector<vector<vector<Ciphertext>>> cts_temp_1(3); //3*16*2

		for (int k=0; k<3; k++){

			cts_conv1_1[k].resize(dimension);

			for(int i=0; i<dimension; i++){

				cts_conv1_1[k][i].resize(2);	

				for (int r=0; r < 2; r++){	

					cts_conv1_1[k][i][r].resize(64);
				}
			}
		}		

		for (int i=0; i<dimension; i++){

			for (int j=0; j < 64; j++){

				Ciphertext temp;
				evaluator.rotate_vector(st_gcn0_output[i], 64*j, galk, temp);

				for (int k=0; k<3; k++){
					for (int r=0; r < 2; r++){
						cts_conv1_1[k][i][r][j]=temp;
					}
				}
			}
		}

		cout<<"st-gcn layer 1 conv1 rotation finished"<<endl;

		for (int k=0; k<3; k++){
			
			cts_temp_1[k].resize(dimension);

			for(int i=0; i<dimension; i++){

				cts_temp_1[k][i].resize(2);				

				for (int r=0; r < 2; r++){

					for (int j=0; j < 64; j++){

						//cts_conv1_1[k][i][r]=conv1_1[i];
						//evaluator.rotate_vector(st_gcn0_output[i], 64*j, galk, cts_conv1_1[k][i][r][j]);
	
						parms_id_type last_parms_id = st_gcn0_output[i].parms_id();
						evaluator.mod_switch_to_inplace(conv1_1_plain[k][r][j], last_parms_id);

						evaluator.multiply_plain_inplace(cts_conv1_1[k][i][r][j],conv1_1_plain[k][r][j]);

					}
					evaluator.add_many(cts_conv1_1[k][i][r],cts_temp_1[k][i][r]);
					evaluator.rescale_to_next_inplace(cts_temp_1[k][i][r]);

					// add bias1
					cts_temp_1[k][i][r].scale() = scale;
					parms_id_type temp_parms_id = cts_temp_1[k][i][r].parms_id();
					evaluator.mod_switch_to_inplace(bias1_1_plain[k][r], temp_parms_id);
					evaluator.add_plain_inplace(cts_temp_1[k][i][r], bias1_1_plain[k][r]);
				}
			}
		}
		
		cout<< "Decrypt the result after the st-gcn layer 1 conv1"<<endl;

		Plaintext output_plain;
		vector<double> output_featuremap;

		decryptor.decrypt(cts_temp_1[0][0][0],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);


		decryptor.decrypt(cts_temp_1[0][0][1],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);


		decryptor.decrypt(cts_temp_1[0][1][0],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);

		decryptor.decrypt(cts_temp_1[0][2][0],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);



		/*times with sparse matrix */
		vector<vector<Ciphertext>> cts_A_1(dimension); //16*2

		vector<vector<vector<Ciphertext>>> cts_preA_1{cts_temp_1[0], cts_temp_1[1], cts_temp_1[1], cts_temp_1[2]}; //4*16*2
		
		for(int i=0; i<dimension; i++){

			cts_A_1[i].resize(2);

			vector<vector<Ciphertext>> matrix_temp(2);

			for(int r=0; r<2; r++){

				for (int v=0; v<4; v++){

					if( index_set[v][i]<16 ){

						parms_id_type last_parms_id = cts_preA_1[v][index_set[v][i]][r].parms_id();
						evaluator.mod_switch_to_inplace(A_plain_0[v][i][r], last_parms_id);

						evaluator.multiply_plain_inplace(cts_preA_1[v][index_set[v][i]][r], A_plain_0[v][i][r]);	
						
						//cout<<"test debugging---------------------------------"<<endl;
						
						matrix_temp[r].emplace_back(move(cts_preA_1[v][index_set[v][i]][r]));

					}
				}

				//cout<<"matrix_temp size "<<matrix_temp[r].size()<<endl;

				evaluator.add_many(matrix_temp[r], cts_A_1[i][r]);
				evaluator.rescale_to_next_inplace(cts_A_1[i][r]);
				//cout<<"scale after conv1 and mult A: "<<cts_A[i].scale()<<endl;
				cts_A_1[i][r].scale()=scale;

				parms_id_type parms_id_b1 = cts_A_1[i][r].parms_id();
				evaluator.mod_switch_to_inplace(b1_1_plain[r], parms_id_b1);
				evaluator.add_plain_inplace(cts_A_1[i][r],b1_1_plain[r]);
				
				// evaluator.square_inplace(cts_A_1[i][r]);
				// evaluator.relinearize_inplace(cts_A_1[i][r],relin_keys);
				// evaluator.rescale_to_next_inplace(cts_A_1[i][r]);
				// cts_A_1[i][r].scale()=scale;

				//cout<<"scale after bn 1: "<<cts_A[i].scale()<<endl;
			}
		}

		cout<< "Decrypt the result after the conv1, A , bn1, x^2"<<endl;
		//Plaintext output_plain;
		//vector<double> output_featuremap;
		decryptor.decrypt(cts_A_1[0][0],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,256);
		
		decryptor.decrypt(cts_A_1[0][1],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,256);

		decryptor.decrypt(cts_A_1[2][0],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap);

		decryptor.decrypt(cts_A_1[15][0],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap);		 


		/* second conv 9*1 layer & bn1 layer & bn2 layer & x^2 activation*/
		vector<vector<Ciphertext>> cts_conv2_1(dimension);

		vector<vector<vector<Ciphertext>>> conv2_1_rotated_intra(dimension); //16*2*9

		vector<vector<vector<vector<vector<Ciphertext>>>>> conv2_1_temp(dimension);

		vector<vector<vector<vector<Ciphertext>>>> conv2_1_rotated_extra(dimension);

		vector<vector<vector<Ciphertext>>> conv2_1_rotated_extra_temp(dimension);


		for (int i=0; i< dimension; i++){

			conv2_1_rotated_intra[i].resize(2);

			for (int j=0; j<2; j++){

				conv2_1_rotated_intra[i][j].resize(9);

				for (int r=0; r<9; r++){

					evaluator.rotate_vector(cts_A_1[i][j], (r-4), galk, conv2_1_rotated_intra[i][j][r]);
				}
			}
		}
		cout<<"baby-step ciphertext rotation end"<<endl;

		for (int i=0; i<dimension; i++){

			cts_conv2_1[i].resize(2);
			conv2_1_temp[i].resize(2);
			conv2_1_rotated_extra[i].resize(2);
			conv2_1_rotated_extra_temp[i].resize(2);

			{
			Stopwatch sw("computation time for one ciphertext");


			for (int m=0; m<2; m++){ 
				
				conv2_1_temp[i][m].resize(2);
				conv2_1_rotated_extra[i][m].resize(2);
				conv2_1_rotated_extra_temp[i][m].resize(2);		

				for (int r=0; r<2 ; r++){

					conv2_1_temp[i][m][r].resize(64);
					conv2_1_rotated_extra[i][m][r].resize(64);
					
					for (int k=0; k < 64; k++){
						
						conv2_1_temp[i][m][r][k].resize(9);

						for (int j=0; j < 9; j++){  

							conv2_1_temp[i][m][r][k][j]=conv2_1_rotated_intra[i][r][j];

							parms_id_type last_parms_id = conv2_1_rotated_intra[i][r][j].parms_id();
							evaluator.mod_switch_to_inplace(conv2_1_plain[m][r][k][j], last_parms_id);

							evaluator.multiply_plain_inplace(conv2_1_temp[i][m][r][k][j], conv2_1_plain[m][r][k][j]);

						}

						evaluator.add_many(conv2_1_temp[i][m][r][k], conv2_1_rotated_extra[i][m][r][k]);
						evaluator.rotate_vector(conv2_1_rotated_extra[i][m][r][k], 64*k, galk, conv2_1_rotated_extra[i][m][r][k]);
					}

					evaluator.add_many(conv2_1_rotated_extra[i][m][r], conv2_1_rotated_extra_temp[i][m][r]);
			

				}
	
				evaluator.add_many(conv2_1_rotated_extra_temp[i][m], cts_conv2_1[i][m]);

				evaluator.rescale_to_next_inplace(cts_conv2_1[i][m]);
				cts_conv2_1[i][m].scale() = scale;

				/*add bias2*/
				parms_id_type temp_parms_id = cts_conv2_1[i][m].parms_id();
				evaluator.mod_switch_to_inplace(bias2_1_plain[m], temp_parms_id);
				evaluator.add_plain_inplace(cts_conv2_1[i][m], bias2_1_plain[m]);

				/*Relu Approximation*/

				/*temporialy remove*/
				
				// evaluator.square_inplace(cts_conv2_1[i][m]);
				// evaluator.relinearize_inplace(cts_conv2_1[i][m],relin_keys);
				// evaluator.rescale_to_next_inplace(cts_conv2_1[i][m]);
				// cts_conv2_1[i][m].scale()=scale;

				//cout<<"scale after bn 2: "<<cts_conv2[i].scale()<<endl;

				/*multiply with mask and pre-processing*/
			}
			}

		}

		cout<< "Decrypt the result after st-gcn layer 1, conv2, bn2, x^2"<<endl;

		decryptor.decrypt(cts_conv2_1[0][0],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);
		
		decryptor.decrypt(cts_conv2_1[1][0],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,8);

		decryptor.decrypt(cts_conv2_1[2][0],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,8);

		decryptor.decrypt(cts_conv2_1[0][1],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,8);	

		decryptor.decrypt(cts_conv2_1[1][1],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,8);	

		decryptor.decrypt(cts_conv2_1[2][1],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,8);	

		cout<<"st-gcn layer 1 computation finished"<<endl;

		for (int i=0; i< dimension;i++){
			
			for (int r=0; r < 2; r++){

				parms_id_type temp_parms_id = cts_conv2_1[i][r].parms_id();
				evaluator.mod_switch_to_inplace(mask_plain, temp_parms_id);
				evaluator.multiply_plain_inplace(cts_conv2_1[i][r], mask_plain);

				evaluator.rotate_vector(cts_conv2_1[i][r], (-1)*r, galk, cts_conv2_1[i][r]);
			}

			evaluator.add_many(cts_conv2_1[i], st_gcn1_output[i]);

			evaluator.rescale_to_next_inplace(st_gcn1_output[i]);
			st_gcn1_output[i].scale() = scale;
		}


		cout<<"after pre-processing for the st-gcn layer 1 result"<<endl;
		decryptor.decrypt(st_gcn1_output[0],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);
		
		decryptor.decrypt(st_gcn1_output[1],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);

		decryptor.decrypt(st_gcn1_output[2],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);

		decryptor.decrypt(st_gcn1_output[3],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);

		//st_gcn1_output=cts_conv2_1;
	}


	/*third st-gcn layer*/
	vector<vector<double>> edge_importance2;

	vector<vector<double>> conv1_2;
	vector<vector<double>> bias1_2;

	vector<double> gamma1_2;
	vector<double> beta1_2;
	vector<double> mean1_2;
	vector<double> var1_2;

	vector<vector<double>> conv2_2;
	vector<double> bias2_2;

	vector<double> gamma2_2;
	vector<double> beta2_2;
	vector<double> mean2_2;
	vector<double> var2_2;


	edge_importance2=read_params_2dim("./params/edge_importance.3.txt");

	conv1_2=read_params_2dim("./params/st_gcn_networks.3.gcn.conv.weight.txt");
	bias1_2=read_params_2dim("./params/st_gcn_networks.3.gcn.conv.bias.txt");

	gamma1_2=read_params_1dim("./params/st_gcn_networks.3.tcn.0.weight.txt");
	beta1_2=read_params_1dim("./params/st_gcn_networks.3.tcn.0.bias.txt");
	mean1_2=read_params_1dim("./params/st_gcn_networks.3.tcn.0.running_mean.txt");
	var1_2=read_params_1dim("./params/st_gcn_networks.3.tcn.0.running_var.txt");

	conv2_2=read_params_2dim("./params/st_gcn_networks.3.tcn.2.weight.txt");
	bias2_2=read_params_1dim("./params/st_gcn_networks.3.tcn.2.bias.txt");

	gamma2_2=read_params_1dim("./params/st_gcn_networks.3.tcn.3.weight.txt");
	beta2_2=read_params_1dim("./params/st_gcn_networks.3.tcn.3.bias.txt");
	mean2_2=read_params_1dim("./params/st_gcn_networks.3.tcn.3.running_mean.txt");
	var2_2=read_params_1dim("./params/st_gcn_networks.3.tcn.3.running_var.txt");


	vector<vector<vector<double>>> reshape_conv1_2;
	vector<vector<vector<double>>> reshape_conv2_2; 

	reshape_conv1_2=reshape_2to3(3,conv1_2);

	int	output_channel_2=reshape_conv1_2[0].size();

	reshape_conv2_2=reshape_2to3(output_channel_2,conv2_2);

	for (int i=0; i<var1_2.size() ;i++){
		var1_2[i]=sqrt(var1_2[i]);
		var2_2[i]=sqrt(var2_2[i]);
	}

	cout<<"current outputchannel size :"<<output_channel_2<<endl;

	/* the st-gcn layer 1 implementation*/
	cout<<"conv1_2 weight layer size "<<reshape_conv1_2.size()<<" , "<<reshape_conv1_2[0].size()<<" , "<<reshape_conv1_2[0][0].size()<<endl;
	cout<<"conv1_2 bias layer size "<<bias1_2.size()<<" , "<<bias1_2[0].size()<<endl;

	cout<<"bn1_2 gamma1 layer size "<<gamma1_2.size()<<endl;
	cout<<"bn1_2 beta1 layer size "<<beta1_2.size()<<endl;
	cout<<"bn1_2 mean1 layer size "<<mean1_2.size()<<endl;
	cout<<"bn1_2 var1 layer size "<<var1_2.size()<<endl;

	cout<<"conv2_2 layer size "<<reshape_conv2_2.size()<<" , "<<reshape_conv2_2[0].size()<<" , "<<reshape_conv2_2[0][0].size()<<endl;
	cout<<"conv2_2 bias layer size "<<bias2_2.size()<<endl;

	cout<<"bn2_2 gamma1 layer size "<<gamma2_2.size()<<endl;
	cout<<"bn2_2 beta1 layer size "<<beta2_2.size()<<endl;
	cout<<"bn2_2 mean1 layer size "<<mean2_2.size()<<endl;
	cout<<"bn2_2 var1 layer size "<<var2_2.size()<<endl;


	// init the square Matrxi A0, A1, A2
	vector<vector<double>> A0_2(dimension);
	vector<vector<double>> A1_2(dimension);
	vector<vector<double>> A2_2(dimension);
	
	for (int i = 0;  i < dimension; i++){
		A0_2[i].resize(dimension);
		A1_2[i].resize(dimension);
		A2_2[i].resize(dimension);
		for (int j = 0; j < dimension; j++ ){
			A0_2[i][j] = raw_A[i][j]*edge_importance2[i][j];
			A1_2[i][j] = raw_A[i+25][j]*edge_importance2[i+25][j];
			A2_2[i][j] = raw_A[i+50][j]*edge_importance2[i+50][j];
		}
		//print_vector(A0[i],3,7);
	}


	/*encode the conv1 weight into plaintexts, first absorb the bn parameters*/
	vector<vector<vector<vector<double>>>> conv1_2_encode(3); 
	//3*2*64*4096

	for (int r=0; r<3 ;r++){

		conv1_2_encode[r].resize(2);

		for (int k=0; k < 2; k++){

			conv1_2_encode[r][k].resize(64);

			for (int v=0; v< 64 ;v++){

				conv1_2_encode[r][k][v].resize(slot_count);

				for(int j=0; j<slot_count; j++){

					conv1_2_encode[r][k][v][j]=reshape_conv1_2[r][ j / 64 + k*64 ][ (j / 64 + v) % 64 + 64 * (j % 2) ];

				}
			}
		}
	}
	cout<<"print conv1_2 encode: "<<endl;
	print_vector(conv1_2_encode[0][0][0],129);

	vector<vector<vector<Plaintext>>> conv1_2_plain(3); // 3*2*64

	for (int r=0; r<3 ;r++){
		conv1_2_plain[r].resize(2);
		for (int k =0; k<2; k++){
			conv1_2_plain[r][k].resize(64);
			for (int v=0; v<64; v++){
				encoder.encode(conv1_2_encode[r][k][v], scale, conv1_2_plain[r][k][v]);
			}
		}
	}

	/*encode the conv bias 1 & 2 into plaintexts*/
	vector<vector<double>> bias1_2_encode(3); //3*2*4096
	vector<Plaintext> bias1_2_plain(3);

	for (int i=0; i<3; i++){

		bias1_2_encode[i].resize(slot_count);

		for (int j=0; j<slot_count; j++){

			bias1_2_encode[i][j]=bias1_2[i][ j / 64 + (j % 2)*64 ];
		}
		
		encoder.encode(bias1_2_encode[i], scale, bias1_2_plain[i]);
	}

	cout<<"print conv1_2 bias encode: "<<endl;
	print_vector(bias1_2_encode[0],128,4);

	//encoding the matrix plaintexts for sparse matrix multiplication
	vector<vector<double>> ffpA0, ffpA1_0, ffpA1_1, ffpA2; //16*4096
	
	ffpA0=matrix_encode(ffpA0, A0_2, xindex0, yindex0, dimension, slot_count);
	ffpA1_0=matrix_encode(ffpA1_0, A1_2, xindex1_0, yindex1_0, dimension, slot_count);
	ffpA1_1=matrix_encode(ffpA1_1, A1_2, xindex1_1, yindex1_1, dimension, slot_count);
	ffpA2=matrix_encode(ffpA2, A2_2, xindex2, yindex2, dimension, slot_count);

	cout<<"The original matrix size "<<ffpA0.size()<<" "<<ffpA0[0].size()<<endl;

	vector<vector<double>> ffpA0m(dimension);     
	vector<vector<double>> ffpA1_0m(dimension);   
	vector<vector<double>> ffpA1_1m(dimension);  
	vector<vector<double>> ffpA2m(dimension);      
	//2*16*4096

	/*encode the A matrix in multiple channels */
	/*absorb the conv1 layer with matrix A and bn1*/

	for (int k=0; k < dimension ;k++){
		
		ffpA0m[k].resize(slot_count);
		ffpA1_0m[k].resize(slot_count);
		ffpA1_1m[k].resize(slot_count);
		ffpA2m[k].resize(slot_count);

		// for (int i=0; i<2 ;i++){

		// 	fpA0m[k][i].resize(slot_count);
		// 	fpA1_0m[k][i].resize(slot_count);
		// 	fpA1_1m[k][i].resize(slot_count);
		// 	fpA2m[k][i].resize(slot_count);

		for (int j=0; j < slot_count; j++){

			double temp_scale=gamma1_2[ j / 64 + (j % 2)*64 ] / var1_2[ j / 64 + (j % 2)*64 ];

			ffpA0m[k][j]  = ffpA0[k][j] * temp_scale;
			ffpA1_0m[k][j]= ffpA1_0[k][j] * temp_scale;
			ffpA1_1m[k][j]= ffpA1_1[k][j] * temp_scale;
			ffpA2m[k][j]  = ffpA2[k][j]   * temp_scale;
		}
	}
	cout<<"matrix A_1: "<<endl;
	print_vector(A0_2[0],16,3);
	cout<<"encode matrix A_1: "<<endl;
	print_vector(ffpA0m[0],16,3);

	//cout<<"The original matrix size "<<fpA0m.size()<<" "<<fpA0m[0].size()<<" "<<fpA0m[0][0].size()<<endl;
	

	/* encode the sparse matrix into plaintexts*/
	
	vector<vector<vector<double>>> A_sparse_2{ffpA0m, ffpA1_0m, ffpA1_1m, ffpA2m}; //4*16*4096

	vector<vector<Plaintext>> A_plain_2(4);

	// 4 * 16  (16 columns) 

	for (int k =0; k<4; k++){

		A_plain_2[k].resize(dimension);

		for(int i=0; i<dimension; i++){

			encoder.encode(A_sparse_2[k][i], scale, A_plain_2[k][i]);
			
		}
	}

	cout<<"matrix size "<<A_plain_2.size()<<" "<<A_plain_2[0].size()<<endl;

	/*abosrb & encode the bn1 bias into plaintext*/
	vector<double> b1_2(slot_count);
	Plaintext b1_2_plain;

	for (int i=0; i< slot_count ;i++){

		b1_2[i]=beta1_2[i / 64 + (i % 2)*64] -mean1_2[ i / 64 + (i % 2)*64 ]*(gamma1_2[ i / 64 + (i % 2)*64 ]/var1_2[ i / 64 + (i % 2)*64 ]);
	}
	encoder.encode(b1_2, scale, b1_2_plain);


	/*absorb bn2 into conv2 weights*/
	for (int k=0; k<reshape_conv2_2.size(); k++){

		for (int i=0; i<reshape_conv2_2[0].size() ;i++){	

			for ( int j=0; j< reshape_conv2_2[0][0].size(); j++){

				reshape_conv2_2[k][i][j]= reshape_conv2_2[k][i][j] * (gamma2_2[k]/var2_2[k]);

			}
		}
	}

	/* encode the conv2 weights into plaintexts*/
	//vector<vector<vector<vector<vector<double>>>>>  conv2_2_encode(2);  // 2*2*64*9*4096
	vector<vector<vector<vector<double>>>>  conv2_2_encode(2);  // 2*64*9*4096
	//reshape_conv2 : 128(output channel) * 128(input channel) * 9
	
	// for (int m=0; m <2 ; m++){

	// 	conv2_2_encode[m].resize(2);

	for (int v=0; v<2; v++){

		//conv2_2_encode[m][v].resize(64);
		conv2_2_encode[v].resize(64);

		for (int i=0; i< 64; i++){

			//conv2_2_encode[m][v][i].resize(9);
			conv2_2_encode[v][i].resize(9);

			for (int r=0; r<9; r++){

				//conv2_2_encode[m][v][i][r].resize(slot_count);
				conv2_2_encode[v][i][r].resize(slot_count);

				for (int j=0; j<slot_count; j++){

					if( (r < 4) & ( ((j % 32) /2) < (4-r)) ){

						//conv2_2_encode[m][v][i][r][j]= 0;
						conv2_2_encode[v][i][r][j]= 0;

					}else if((r > 4) & ( ((j % 32) /2) >= (16-(r-4))) ){
						
						//conv2_2_encode[m][v][i][r][j]= 0;
						conv2_2_encode[v][i][r][j]= 0;

					}else{

						//conv2_2_encode[m][v][i][r][j]= reshape_conv2_2[ (((j / 64) + (64-i)) % 64)+64*m  ][ (j / 64) + 64*v ][r];
						conv2_2_encode[v][i][r][j]= reshape_conv2_2[ (((j / 64) + (64-i)) % 64)+64*v  ][ (j / 64) + (j % 2)*64 ][r];
						
					}
				}						
			}
		}
	}
	
	cout<<"print conv2_2 encode: "<<endl;
	print_vector(conv2_2_encode[0][0][0],128);
	//print_vector(conv2_1_encode[1][0][0],32);
	//print_vector(conv2_1_encode[2][0][0],32);

	/*encode the conv2 weight into plaintexts*/

	//vector<vector<vector<vector<Plaintext>>>> conv2_2_plain(2);//2*2*64*9
	vector<vector<vector<Plaintext>>> conv2_2_plain(2);//2*64*9

	//int stat_time=0;
	// for (int m=0; m<2; m++){

	// 	conv2_2_plain[m].resize(2);

	for(int k=0; k<2; k++){

		//conv2_2_plain[m][k].resize(64);
		conv2_2_plain[k].resize(64);

		for (int i=0; i< 64; i++){

			//conv2_2_plain[m][k][i].resize(9);
			conv2_2_plain[k][i].resize(9);

			for (int j=0; j<9; j++){
			
				//encoder.encode(conv2_2_encode[m][k][i][j],scale,conv2_2_plain[m][k][i][j]);
				encoder.encode(conv2_2_encode[k][i][j],scale,conv2_2_plain[k][i][j]);
			}
		}
	}
	

	/*encode the conv bias 2 into plaintexts*/ /*absorb the bn2 bias with conv2 layer bias */

	for (int i=0; i<bias2_2.size() ;i++){
		bias2_2[i]= beta2_2[i]-(gamma2_2[i]/var2_2[i])*mean2_2[i] + (gamma2_2[i]/var2_2[i]) * bias2_2[i];
	}

	// vector<vector<double>> bias2_2_encode(2);
	// vector<Plaintext> bias2_2_plain(2);

	// for(int v=0; v< 2; v++){

	// 	bias2_2_encode[v].resize(slot_count);

	// 	for (int i=0; i<slot_count; i++){

	// 		bias2_2_encode[v][i] = bias2_2[ (slot_count*v + i) / 64 ];
	// 	}
	// 	encoder.encode(bias2_2_encode[v], scale, bias2_2_plain[v]);
	// }

	vector<double> bias2_2_encode(slot_count);
	Plaintext bias2_2_plain;

	for (int i=0; i<slot_count; i++){

		bias2_2_encode[i] = bias2_2[ i / 64 + (i % 2)*64 ];
	}
	encoder.encode(bias2_2_encode, scale, bias2_2_plain);


	vector<Ciphertext> st_gcn2_output;
	{
		Stopwatch sw("Inference time for st-gcn layer 2");
		/*start the inference for st-gcn layer 1*/
		//vector<vector<Ciphertext>> cts_conv1_1{cts_conv2,cts_conv2,cts_conv2};
		//vector<vector<vector<Ciphertext>>> cts_rotated(3);

		vector<vector<vector<vector<Ciphertext>>>> cts_conv1_2(3); // 3*16*2*64
		vector<vector<vector<Ciphertext>>> cts_temp_2(3); //3*16*2
		vector<vector<Ciphertext>> cts_conv1_temp(3); //3*16

		for (int k=0; k<3; k++){

			cts_conv1_2[k].resize(dimension);

			for(int i=0; i<dimension; i++){

				cts_conv1_2[k][i].resize(2);	

				for (int r=0; r < 2; r++){	

					cts_conv1_2[k][i][r].resize(64);
				}
			}
		}		

		for (int i=0; i<dimension; i++){

			for (int j=0; j < 64; j++){

				Ciphertext temp;
				evaluator.rotate_vector(st_gcn1_output[i], 64*j, galk, temp);

				for (int k=0; k<3; k++){
					for (int r=0; r < 2; r++){
						cts_conv1_2[k][i][r][j]=temp;
					}
				}
			}
		}

		cout<<"st-gcn layer 2 conv1 rotation finished"<<endl;

		for (int k=0; k<3; k++){
			
			cts_temp_2[k].resize(dimension);
			cts_conv1_temp[k].resize(dimension);

			for(int i=0; i<dimension; i++){

				cts_temp_2[k][i].resize(2);				

				for (int r=0; r < 2; r++){

					for (int j=0; j < 64; j++){
	
						parms_id_type last_parms_id = st_gcn1_output[i].parms_id();
						evaluator.mod_switch_to_inplace(conv1_2_plain[k][r][j], last_parms_id);

						evaluator.multiply_plain_inplace(cts_conv1_2[k][i][r][j],conv1_2_plain[k][r][j]);
					}

					evaluator.add_many(cts_conv1_2[k][i][r], cts_temp_2[k][i][r]);

					evaluator.rescale_to_next_inplace(cts_temp_2[k][i][r]);
					cts_temp_2[k][i][r].scale() = scale;

					Ciphertext rotated_temp;
					evaluator.rotate_vector(cts_temp_2[k][i][r] , 1 , galk, rotated_temp);
					evaluator.add(cts_temp_2[k][i][r], rotated_temp, cts_temp_2[k][i][r]);

					parms_id_type temp_parms_id = cts_temp_2[k][i][r].parms_id();
					evaluator.mod_switch_to_inplace(mask_plain, temp_parms_id);
					evaluator.multiply_plain_inplace(cts_temp_2[k][i][r], mask_plain);

					evaluator.rotate_vector(cts_temp_2[k][i][r], (-1)*r, galk, cts_temp_2[k][i][r]);
				}
				
				evaluator.add_many(cts_temp_2[k][i],cts_conv1_temp[k][i]);

				evaluator.rescale_to_next_inplace(cts_conv1_temp[k][i]);
				cts_conv1_temp[k][i].scale() = scale;

				// add bias1
				parms_id_type temp_parms_id = cts_conv1_temp[k][i].parms_id();
				evaluator.mod_switch_to_inplace(bias1_2_plain[k], temp_parms_id);

				evaluator.add_plain_inplace(cts_conv1_temp[k][i], bias1_2_plain[k]);
			}
		}

		
		cout<< "Decrypt the result after the st-gcn layer 2 conv1"<<endl;

		Plaintext output_plain;
		vector<double> output_featuremap;

		decryptor.decrypt(cts_conv1_temp[0][0],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);


		decryptor.decrypt(cts_conv1_temp[0][1],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);


		decryptor.decrypt(cts_conv1_temp[0][2],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);

		decryptor.decrypt(cts_conv1_temp[0][3],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);


		/*times with sparse matrix */
		vector<Ciphertext> cts_A_2(dimension); //16
		vector<vector<Ciphertext>> cts_preA_2{cts_conv1_temp[0], cts_conv1_temp[1], cts_conv1_temp[1], cts_conv1_temp[2]}; //4*16
		
		for(int i=0; i<dimension; i++){

			vector<Ciphertext> matrix_temp;

			for (int v=0; v<4; v++){

				if( index_set[v][i]<16 ){

					parms_id_type last_parms_id = cts_preA_2[v][index_set[v][i]].parms_id();
					evaluator.mod_switch_to_inplace(A_plain_2[v][i], last_parms_id);

					evaluator.multiply_plain_inplace(cts_preA_2[v][index_set[v][i]], A_plain_2[v][i]);	
					matrix_temp.emplace_back(move(cts_preA_2[v][index_set[v][i]]));
				}
			}
			//cout<<"matrix_temp size "<<matrix_temp[r].size()<<endl;
			evaluator.add_many(matrix_temp, cts_A_2[i]);
			evaluator.rescale_to_next_inplace(cts_A_2[i]);
			cts_A_2[i].scale()=scale;

			parms_id_type parms_id_b1 = cts_A_2[i].parms_id();
			evaluator.mod_switch_to_inplace(b1_2_plain, parms_id_b1);
			evaluator.add_plain_inplace(cts_A_2[i],b1_2_plain);
			

			/*temporarily remove*/
			// evaluator.square_inplace(cts_A_2[i]);
			// evaluator.relinearize_inplace(cts_A_2[i],relin_keys);
			// evaluator.rescale_to_next_inplace(cts_A_2[i]);

			//cout<<"scale after bn 1: "<<cts_A[i].scale()<<endl;
		}

		cout<< "Decrypt the result after the conv1, A , bn1, x^2"<<endl;
		//Plaintext output_plain;
		//vector<double> output_featuremap;
		decryptor.decrypt(cts_A_2[0],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);
		
		decryptor.decrypt(cts_A_2[1],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);

		decryptor.decrypt(cts_A_2[2],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);

		decryptor.decrypt(cts_A_2[3],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);		 


		/* second conv 9*1 layer & bn1 layer & bn2 layer & x^2 activation*/

		vector<vector<Ciphertext>> cts_conv2_2(dimension);
		vector<vector<Ciphertext>> conv2_2_rotated_intra(dimension); //16*9
		vector<vector<vector<vector<Ciphertext>>>> conv2_2_temp(dimension); //16*2*64*9
		vector<vector<vector<Ciphertext>>> conv2_2_rotated_extra(dimension); //16*2*64

		vector<Ciphertext> cts_conv2_2_final(dimension);
		//vector<vector<Ciphertext>> conv2_2_rotated_extra_temp(dimension);

		for (int i=0; i< dimension; i++){

			conv2_2_rotated_intra[i].resize(9);

			for (int r=0; r<9; r++){

				evaluator.rotate_vector(cts_A_2[i], (r-4)*2, galk, conv2_2_rotated_intra[i][r]);
			}
		}
		cout<<"st-gcn layer 2 conv2 baby-step ciphertext rotation end"<<endl;

		for (int i=0; i<dimension; i++){

			cts_conv2_2[i].resize(2);
			conv2_2_temp[i].resize(2);
			conv2_2_rotated_extra[i].resize(2);

			//conv2_2_rotated_extra_temp[i].resize(2);

			{
			Stopwatch sw("computation time for one ciphertext");
			for (int r=0; r<2 ; r++){

				conv2_2_temp[i][r].resize(64);
				conv2_2_rotated_extra[i][r].resize(64);
				
				for (int k=0; k < 64; k++){
					
					conv2_2_temp[i][r][k].resize(9);

					for (int j=0; j < 9; j++){  

						conv2_2_temp[i][r][k][j]=conv2_2_rotated_intra[i][j];
						parms_id_type last_parms_id = conv2_2_rotated_intra[i][j].parms_id();
						evaluator.mod_switch_to_inplace(conv2_2_plain[r][k][j], last_parms_id);

						evaluator.multiply_plain_inplace(conv2_2_temp[i][r][k][j], conv2_2_plain[r][k][j]);

					}
					evaluator.add_many(conv2_2_temp[i][r][k], conv2_2_rotated_extra[i][r][k]);
					evaluator.rotate_vector(conv2_2_rotated_extra[i][r][k], 64*k, galk, conv2_2_rotated_extra[i][r][k]);
				}
				evaluator.add_many(conv2_2_rotated_extra[i][r], cts_conv2_2[i][r]);

				evaluator.rescale_to_next_inplace(cts_conv2_2[i][r]);
				cts_conv2_2[i][r].scale() = scale;

				Ciphertext rotated_temp;
				evaluator.rotate_vector( cts_conv2_2[i][r], 1 , galk, rotated_temp);

				evaluator.add(cts_conv2_2[i][r] , rotated_temp, cts_conv2_2[i][r]);

				parms_id_type parms_id = cts_conv2_2[i][r].parms_id();
				evaluator.mod_switch_to_inplace(mask_plain, parms_id);

				evaluator.multiply_plain_inplace(cts_conv2_2[i][r], mask_plain);
				evaluator.rotate_vector(cts_conv2_2[i][r], (-1)*r ,galk,cts_conv2_2[i][r]);
			}

			evaluator.add_many(cts_conv2_2[i], cts_conv2_2_final[i]);

			evaluator.rescale_to_next_inplace(cts_conv2_2_final[i]);
			cts_conv2_2_final[i].scale() = scale;

			/*add st-gcn layer 3 bias2*/
			parms_id_type temp_parms_id = cts_conv2_2_final[i].parms_id();
			evaluator.mod_switch_to_inplace(bias2_2_plain, temp_parms_id);

			evaluator.add_plain_inplace(cts_conv2_2_final[i], bias2_2_plain);

			/*temporaily remove*/
			/*Relu Approximation*/
			// evaluator.square_inplace(cts_conv2_1[i][m]);
			// evaluator.relinearize_inplace(cts_conv2_1[i][m],relin_keys);
			// evaluator.rescale_to_next_inplace(cts_conv2_1[i][m]);
			// cts_conv2_1[i][m].scale()=scale;

			//cout<<"scale after bn 2: "<<cts_conv2[i].scale()<<endl;
			}

		}

		cout<< "Decrypt the result after st-gcn layer 2, conv2, bn2, x^2"<<endl;

		decryptor.decrypt(cts_conv2_2_final[0],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);
		
		decryptor.decrypt(cts_conv2_2_final[1],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);

		decryptor.decrypt(cts_conv2_2_final[2],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);

		decryptor.decrypt(cts_conv2_2_final[3],output_plain);
		encoder.decode(output_plain,output_featuremap);
		print_vector(output_featuremap,128);		

		cout<<"st-gcn layer 2 computation finished"<<endl;

		st_gcn2_output=cts_conv2_2_final;

	}	

	/*global average pooling & Fully connected layer*/

	vector<vector<double>> fc_weight;
	vector<double> fc_bias;

	fc_weight=read_params_2dim("./params/fcn.weight.txt");
	fc_bias=read_params_1dim("./params/fcn.bias.txt");

	for (int i=0; i< fc_weight.size() ;i++){
		fc_weight[i].resize(128);
	}

	cout<<"fc layer weight size "<<fc_weight.size()<<" "<<fc_weight[0].size()<<endl;
	cout<<"fc layer bias size "<<fc_bias.size()<<endl;
	
	Ciphertext col_sum;
	evaluator.add_many(st_gcn2_output, col_sum);

	{
	Plaintext output_plain;
	vector<double> output_featuremap;

	decryptor.decrypt(col_sum,output_plain);
	encoder.decode(output_plain,output_featuremap);
	cout<<"col sum"<<endl;
	print_vector(output_featuremap,128);
	}


	Ciphertext person_sum, col_sum_rotated;
	evaluator.rotate_vector( col_sum, 32, galk, col_sum_rotated);
	evaluator.add(col_sum , col_sum_rotated , person_sum);


	{
	Plaintext output_plain;
	vector<double> output_featuremap;

	decryptor.decrypt(person_sum,output_plain);
	encoder.decode(output_plain,output_featuremap);
	cout<<"person sum"<<endl;
	print_vector(output_featuremap,32);
	}

	Ciphertext row_sum;
	row_sum=person_sum;

	for (int i=0; i < 4; i++){
		
		int rotate_number=pow(2,4-i);
		Ciphertext temp;

		evaluator.rotate_vector( row_sum, rotate_number , galk , temp);
		evaluator.add(row_sum, temp , row_sum);


	}

	{
	Plaintext output_plain;
	vector<double> output_featuremap;

	decryptor.decrypt(row_sum,output_plain);
	encoder.decode(output_plain,output_featuremap);
	print_vector(output_featuremap,128);
	}


	vector<double> fc_weight_encode(slot_count,0);
	Plaintext fc_weight_plain;

	for (int j=0; j<slot_count; j++){

		if ( (j % 64) / 2 ==0){

			fc_weight_encode[j]= fc_weight[0][ (j / 64) + (j % 64)*64 ];
		}

	}
	encoder.encode(fc_weight_encode, scale ,fc_weight_plain);

	parms_id_type final_parms_id = row_sum.parms_id();
	evaluator.mod_switch_to_inplace(fc_weight_plain, final_parms_id);

	evaluator.multiply_plain_inplace(row_sum, fc_weight_plain);

	Ciphertext final_result;
	final_result=row_sum;

	for (int i=0; i<6;i++){

		Ciphertext temp;
		int rotate_number= slot_count/ pow(2, (i+1));

		evaluator.rotate_vector( final_result, rotate_number , galk, temp);
		evaluator.add(final_result, temp, final_result);
	}
	
	{
	Plaintext output_plain;
	vector<double> output_featuremap;

	decryptor.decrypt(final_result,output_plain);
	encoder.decode(output_plain,output_featuremap);
	print_vector(output_featuremap,128);
	}

}

