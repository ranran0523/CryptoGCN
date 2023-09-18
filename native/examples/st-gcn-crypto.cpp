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
		cout << name_ << ": " << duration.count() << " milliseconds" << endl;
	}
	
private:
	string name_;
	chrono::high_resolution_clock::time_point start_time_;
};



vector<vector<double>> read_params_2d(string path)
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
		is.close();
	return result;
}

vector<double> read_params_1d(string path)
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

vector<vector<vector<double>>> reshape_2dto3d(int dim1, vector<vector<double>> X)
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

void bootcamp_demo()
{
	// CLIENT'S VIEW
	// Vector of inputs
	size_t nodes= 25;
	size_t frames=300;
	size_t features=3;
	size_t partitions=3;

	int input_channel=3;
	int output_channel=64;

	// missing the encode part

	//missing the encrypted part

	//conv_result=stgcn(input_channel,output_channel,input_featuremap, normalized_A, w1, w2, b1);
	
	// Matrix multiplication part

	//vector<vector<vector<vector<double>>>> conv_result(2,vector<vector<vector<double>>>(256, vector<vector<double>>(75, vector<double>(25, 0))));

	//average pooling
	
	vector<vector<double>> fc;
	fc=read_params_2d("./native/params/fcn.weight.txt");

	vector<double>fc_bias;
	fc_bias=read_params_1d("./native/params/fcn.bias.txt");


	vector<double>avg_pool(conv_result[0].size());

	for(int o=0; o<conv_result[0].size();o++){
		for(int m=0; m< conv_result.size();m++){
			for(int j=0; j< conv_result[0][0].size();j++){
				for(int w=0; w< conv_result[0][0][0].size();w++){
					avg_pool[o] += conv_result[m][o][j][w] / ( conv_result[0][0].size() * conv_result[0][0][0].size() * conv_result.size());
				}
			}
		}
	}

	vector<double>result(fc.size());
	for(int o=0; o<fc.size();o++){
		for(int m=0; m<avg_pool.size();m++){
			result[o] += avg_pool[m]*fc[o][m];
		}
		result[o]+=fc_bias[o];
	}

	
	ofstream outputfile;
    outputfile.open ("final_result.txt");
    for (int p=0; p<result.size();p++){							
		outputfile<<result[p]<<endl;
	}

	outputfile.close();



	

	// Setting up encryption parameters


	vector<double>fc_bias;
	fc_bias=read_params_1d("./native/params/fcn.bias.txt");


	EncryptionParameters parms(scheme_type::CKKS);
	
	size_t poly_modulus_degree = 8192;
	
	parms.set_poly_modulus_degree(poly_modulus_degree);
	int my_scale = 30;
	int last_plain_scale = 20;
	
	parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, my_scale, my_scale, 60 })); 
	// this works with 8192, refer to the standard v2.3.1
	
	// Set up the SEALContext
	auto context = SEALContext::Create(parms);

	cout << "poly_modulus_degree: " << poly_modulus_degree << endl;

	cout << "Parameters are valid: " << boolalpha
	<< context->key_context_data()->qualifiers().parameters_set << endl;
	cout << "Maximal allowed coeff_modulus bit-count for this poly_modulus_degree: "
	<< CoeffModulus::MaxBitCount(poly_modulus_degree) << endl;
	cout << "Current coeff_modulus bit-count: "
	<< context->key_context_data()->total_coeff_modulus_bit_count() << endl;
	
	// Use a scale of 2^30 to encode
	double scale = pow(2.0, my_scale);
	
	// Create a vector of plaintexts
	CKKSEncoder encoder(context);

	//Plaintext pt;
	//encoder.encode(inputs, scale, pt);
	
	// repeat the v.
	Plaintext ptxt_vec;
	vector<double> vrep(encoder.slot_count());
	for (int i = 0; i < vrep.size(); i++) vrep[i] = inputs[i % inputs.size()];
	encoder.encode(vrep, scale, ptxt_vec);
	
	
	// Set up keys
	KeyGenerator keygen(context);
	auto sk = keygen.secret_key();
	auto pk = keygen.public_key();
	auto relin_keys = keygen.relin_keys();
	
	GaloisKeys galk;
	// Create rotation (Galois) keys
	{
		Stopwatch sw("GaloisKeys creation time");

		//Default Galois keys
		//galk = keygen.galois_keys();

		// Optimized Galois keys (only need rotations that are powers of 2 and 1 -14 ) [apparently don't need to ask for 5, 9, 10 ...?]
        //vector<int> rots{ -4, -2, -1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 };
        vector<int> rots{ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 };
		galk = keygen.galois_keys(rots);

		ofstream fs("test.galk", ios::binary);
		keygen.galois_keys_save(rots, fs);
	}

	
	bool encrypt_symmetric = false; 
	// Create ciphertext
	Ciphertext ct;

	if (encrypt_symmetric){ 
		// Set up Encryptor
		Encryptor encryptor(context, sk); // encrypt using secret key


		
		{
			Stopwatch sw("Encryption time");
			encryptor.encrypt_symmetric(ptxt_vec, ct);
			

			ofstream fs("test_c_to_s.ct", ios::binary);
			encryptor.encrypt_symmetric_save(ptxt_vec, fs);
			
		}
	}else {
		// Set up Encryptor
		Encryptor encryptor(context, pk);
	

		{
			Stopwatch sw("Encryption time");
			encryptor.encrypt(ptxt_vec, ct);
		}
		
		// Save to see size
		{
			ofstream fs("test.ct", ios::binary);
			ct.save(fs);
		}

	}



	// Now send this vector to the server!
	// Also save and send the EncryptionParameters.
	
	// SERVER'S VIEW
	
	Evaluator evaluator(context);
	
	// Load EncryptionParameters and set up SEALContext
	
	vector<double> W1{
		-0.025333336, 0.073584445, -0.047216613, 0.18591905, -0.27542248, -0.26681134, -0.13765217, -0.30428806, -0.038901206, 0.25725207, -0.5916338, 0.09984007, 0.1206389, 0.3703078, 0.15509297,
		-0.55778956, -0.17159624, 0.38206482, 0.13647152, 0.0995081, -0.3261224, 0.2189282, -0.16601436, -0.67777795, -0.1130371, 0.3754611, 0.44521803, 0.72404045, -0.1283232, -0.2106342,
		-0.2291026, -0.28999028, -0.074098386, 0.04094153, -0.042301573, 0.25317937, 0.37873283, 0.2503846, -0.4239971, -0.21063337, 0.2852935, 0.097197294, 0.3400989, -0.18595406, 0.16901167,
		0.2522885, 0.4970716, 0.37312597, 0.048000038, 0.5327202, 0.44213235, 0.530003, 0.15933378, 0.33316433, -0.027760876, -0.43822733, 0.33508095, -1.0097011, 0.17022005, 0.19332814,
		0.09802045, 0.34931612, 0.060469475, -0.16971138, -0.123592965, 0.18276687, -0.058420308, 0.26680708, -0.13068433, -0.14461695, -0.017179696, 0.011078397, 0.21604314, -0.27519697, 0.0110200895,
		0.13122348, -0.059484687, 0.096138574, -0.29075062, 0.15073052, 0.42238045, -0.47537068, -0.10458778, -0.3407887, -0.024293313, -0.38392445, 0.0989361, -0.22327657, -0.44290423, 0.59593815,
		-0.30866915, 0.06554472, 0.38248485, -0.26349783, -0.14231941, 0.13525532, -0.3736084, 0.022258941, -0.2930561, -0.3102283, 0.23081024, 0.47965708, 0.4476767, -0.2579037, -0.33573648,
		0.28718016, 0.10692868, -0.0071940785, -0.8159498, 0.25386295, -0.3442934, 0.29050693, -0.41159427, 0.006808188, 0.22295325, -0.8941154, -0.18905659, 0.68109876, 0.28735676, 0.57250553,
		0.15508257, -0.44993335, -0.0030782288, 0.47826147, 0.03962607, -0.011574908, 0.19096713, -0.22766003, 0.059436586, -0.45667365, 0.14223288, -0.45295087, 0.22760722, -0.16525385, -0.2809916,
		0.021617135, 0.06444773, 0.03918349, -0.5673659, 0.8547465, 0.39732796, 0.33035442, -0.21018693, -0.2444701, 0.14366002, 0.16905534, 0.30867127, -0.6301036, 0.32456818, -0.2582209,
		0.4032922, 0.08884215, -0.06644534, 0.16699135, 0.34281743, -0.2986736, -0.35754183, 0.17512605, 0.014519709, 0.34727925, -0.24234116, -0.08042996, -0.2938454, 0.057571694, -0.27575392,
		-0.132408, -0.37733278, -0.20398894, 0.04746109, -0.6032878, 0.109729335, -0.1519245, -0.046319053, -0.30876762, -0.5107319, 0.025014566, -0.11452477, -0.14678663, 0.04716572, 0.5463345,
		0.2238921, 0.43399185, 0.5753054, -0.18898354, 0.13639876, -0.15816061, -0.709152, -0.6216493, -0.32722148, -0.41932252, -0.043735396, 0.20394823, -0.5601734, -0.159232, 0.37627903,
		-0.3102346, 0.3860459, -0.28035825, -0.41002125, -0.11171717, 0.15488632, -0.027155733, 0.42862874, -0.3164006, -0.08545141, 0.16562197, -0.1821526, -0.17811409, -0.2897358, -0.057849944,
		-0.022295434, -0.1930951, -0.46937007, 0.32667413, 0.08007005, 0.32981986, 0.36812696, -0.30101386, -0.7164563, -0.36006156, -0.0005819032, 0.12393838, -0.068327844, 0.49282062, 0.60623616 
	};

	vector<double> W2{
		0.26412928, -0.72620237, -0.2765369, 0.21038327, -0.2737655, 0.2065479, -0.044716638, 0.55226415, 0.27009615, 0.15876967, -0.4926302, 0.14893562, 0.69126004, -0.051137313, -0.6713028
	};
	
	vector<double> b1{
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	};

	double b2 = 0;

	//vector<Plaintext> b1_plaintext(dimension);

	Plaintext b1_plaintext;
	Plaintext b2_plaintext;

	/*
	for (k = 0; k < dimension; k++){
		encoder.encode(b1[k], scale, b1_plaintext[k]);
	}
	*/

	
	

	
	vector<vector<double> > M(dimension);

	int k = 0;

	for (int i = 0;  i < M.size(); i++){
		M[i].resize(dimension);
		for (int j = 0; j < dimension; j++ ){
			M[i][j] = W1[k];
			k++;
		}
	}

	//mult in plaintxt to check result
	vector<double> Mv(dimension,0);
	for (int i = 0;  i < M.size(); i++){
		for (int j = 0; j < dimension; j++){
			Mv[i] += M[i][j] * inputs[j];

		}
	}

	for (int j = 0; j < dimension; j++){
		Mv[j] = Mv[j] + b1[j];
		//cout << "actual: " << Mv[j]  << endl;
	}

	for (int j = 0; j < dimension; j++){
		Mv[j] = Mv[j]*Mv[j];
		//cout << "actual: " << Mv[j]  << endl;
	}

	double sum = 0.0;
	for (int j = 0; j < dimension; j++){
		Mv[j] = Mv[j] * W2[j];
		//cout << "actual: " << Mv[j]  << endl;
		sum += Mv[j];
 	}
 	sum = sum + b2;
	cout << 'Plaintext Multiplication Result' << sum <<endl
	



	




// Encode the diagonals
vector<Plaintext> ptxt_diag(dimension);
for (int i = 0; i < dimension; i++){
	vector<double> diag(dimension);
	for (int j = 0; j < dimension; j++){
		diag[j] = M[j][(j+i) % dimension];
	}
	encoder.encode(diag, scale, ptxt_diag[i]);
}


// Now: perform the multiplication
	Ciphertext temp, temp2;
	Ciphertext enc_result;
	temp2 = ct; 

{

	Stopwatch sw("Homomorphic circuit:");
	//cout << "dimension" << dimension << endl; 
	
	for (int i =0; i < dimension ; i++){
		// rotate
		//cout << "dim " << i << endl;
		//evaluator.rotate_vector(ct, i, galk, temp);
		
		temp = temp2; 
		// multiply
		evaluator.multiply_plain_inplace(temp, ptxt_diag[i]);
		if (i == 0){
			enc_result = temp;
		}else{
			evaluator.add_inplace(enc_result, temp);
		}
		evaluator.rotate_vector(temp2, 1, galk, temp2);
	}
	

	/*
	for (int i =0; i < dimension ; i++){
		// rotate
		//cout << "dim " << i << endl;
		evaluator.rotate_vector(ct, i, galk, temp);
		
		// multiply
		evaluator.multiply_plain_inplace(temp, ptxt_diag[i]);
		if (i == 0){
			enc_result = temp;
		}else{
			evaluator.add_inplace(enc_result, temp);
		}
		
	}*/
	
	//evaluator.relinearize_inplace(enc_result, relin_keys);
	//evaluator.add_plain_inplace(enc_result,b2_plaintext);

	evaluator.rescale_to_next_inplace(enc_result);
	enc_result.scale() = pow(2.0, my_scale);
	
	
	//done with (1) (matrix vector mult)
	// now add b1 bias vector

	/*
	for (k = 0; k < dimension; k++){
		encoder.encode(b1[k],enc_result.parms_id(), scale, b1_plaintext[k]);
	}
	*/

	//batch_encoder.encode(b1,enc_result.parms_id(), scale, b1_plaintext);

	encoder.encode(b1, enc_result.parms_id(),scale, b1_plaintext);
	evaluator.add_plain_inplace(enc_result,b1_plaintext);
	
	evaluator.square(enc_result, enc_result);
	evaluator.relinearize_inplace(enc_result, relin_keys);
	evaluator.rescale_to_next_inplace(enc_result);
	enc_result.scale() = pow(2.0, my_scale);
	
	//done with (2) (square in place)
	
	Plaintext W2_pt; //weight_pt;
	
	encoder.encode(W2, enc_result.parms_id(), pow(2.0, last_plain_scale), W2_pt);

	//Stopwatch sw("Multiply-plain and rescale time");
	evaluator.multiply_plain_inplace(enc_result, W2_pt);
    //evaluator.rescale_to_next_inplace(enc_result);
	
	// Sum the slots
	{
		//Stopwatch sw("Sum-the-slots time");
		Ciphertext temp_ct;
		for (size_t i = 1; i <= encoder.slot_count() / 2; i <<= 1) {
			
			evaluator.rotate_vector(enc_result, i, galk, temp_ct);
			evaluator.add_inplace(enc_result, temp_ct);
		}
	}

	// add bias value b2
	encoder.encode(b2,enc_result.parms_id(), enc_result.scale() , b2_plaintext);
	evaluator.add_plain_inplace(enc_result,b2_plaintext);
   
} //End timing Homomorphic Circuit



//  Back to CLIENT VIEW
 	{
		ofstream fs("test_s_to_c.ct", ios::binary);
        enc_result.save(fs);
    }
	
Plaintext plain_result;
vector<double> result;
Decryptor decryptor(context, sk);

decryptor.decrypt(enc_result, plain_result);
encoder.decode(plain_result, result);

	cout << "Client app gets result: " << result[0] << ", expected: " << sum << endl;

}