#include <iostream>
#include <ctime>
#include <xmmintrin.h>
#include <immintrin.h>

#define N 4096
#define J 233
#define I 199
using namespace std;


void trans(double *x, double *y, size_t n)
{
	double *s = x, *d = y;
	for(int i=0; i<n; i++){
		d = y+i;
		for(int j=0; j<n; j++){
			*d = *(s++);
			d+=n;
		}	
	}
}

/*
假设4x4子块是
a0 a1 a2 a3
b0 b1 b2 b3
c0 c1 c2 c3
d0 d1 d2 d3

目标子块为
a0 b0 c0 d0
a1 b1 c1 d1
a2 b2 c3 d2
a3 b3 c3 d3

*/

void trans_avx256_kernel4x4(const __m256d s1, const __m256d s2, const __m256d s3, const __m256d s4, double *d1, double *d2, double *d3, double *d4)
{
	__m256d t1, t2, t3, t4, t5, t6, t7, t8;
	t1 = _mm256_permute4x64_pd(s1, 0b01001110); //第一行数据进行交换得到a2,a3,a0,a1
	t2 = _mm256_permute4x64_pd(s2, 0b01001110); //第二行数据进行交换得到b2,b3,b0,b1
	t3 = _mm256_permute4x64_pd(s3, 0b01001110); //第三行数据进行交换得到c2,c3,c0,c1
	t4 = _mm256_permute4x64_pd(s4, 0b01001110); //第四行数据进行交换得到d2,d3,d0,d1
	t5 = _mm256_blend_pd(s1, t3, 0b1100); //合并原序列第一行和重排序列第三行得到a0,a1,c0,c1
	t6 = _mm256_blend_pd(s2, t4, 0b1100); //b0,b1,d0,d1
	t7 = _mm256_blend_pd(t1, s3, 0b1100); //a2,a3,c2,c2
	t8 = _mm256_blend_pd(t2, s4, 0b1100); //b2,b3,d2,d3
	/*
	_mm256_store_pd(d1, _mm256_unpacklo_pd(t5,t6));	//生成转置子块，并写入对应位置a0,b0,c0,d0
	_mm256_store_pd(d2, _mm256_unpackhi_pd(t5,t6));	//a1,b1,c1,d1
	_mm256_store_pd(d3, _mm256_unpacklo_pd(t7,t8));	//a2,b2,c2,d2
	_mm256_store_pd(d4, _mm256_unpackhi_pd(t7,t8));	//a3,b3,c3,d3
	*/

	//_mm256_steam_pd 替换_mm256_store_pd, _mm256_steam_pd 将ymm寄存器的数据写入主存时会绕过Cache

	_mm256_stream_pd(d1, _mm256_unpacklo_pd(t5,t6));	//生成转置子块，并写入对应位置a0,b0,c0,d0
	_mm256_stream_pd(d2, _mm256_unpackhi_pd(t5,t6));	//a1,b1,c1,d1
	_mm256_stream_pd(d3, _mm256_unpacklo_pd(t7,t8));	//a2,b2,c2,d2
	_mm256_stream_pd(d4, _mm256_unpackhi_pd(t7,t8));	//a3,b3,c3,d3

}


/*
假设8x8子块是
a0 a1 a2 a3 a4 a5 a6 a7
b0 b1 b2 b3 b4 b5 b6 b7
c0 c1 c2 c3 c4 c5 c6 c7
d0 d1 d2 d3 d4 d5 d6 d7
e0 e1 e2 e3 e4 e5 e6 e7
f0 f1 f2 f3 f4 f5 f6 f7
g0 g1 g2 g3 g4 g5 g6 g7
h0 h1 h2 h3 h4 h5 h6 h7

目标子块为
a0 b0 c0 d0 e0 f0 g0 h0
a1 b1 c1 d1 e1 f1 g1 h1
a2 b2 c3 d2 e2 f2 g2 h2
a3 b3 c3 d3 e3 f3 g3 h3
a4 b4 c4 d4 e4 f4 g4 h4
a5 b5 c5 d5 e5 f5 g5 h5
a6 b6 c6 d6 e6 f6 g6 h6
a7 b7 c7 d7 e7 f7 g7 h7


*/
void trans_avx256_kernel8x8(const double *s1, const double *s2, const double *s3, const double *s4, 
							const double *s5, const double *s6, const double *s7, const double *s8, 
							double *d1, double *d2, double *d3, double *d4,
							double *d5, double *d6, double *d7, double *d8)
{
	__m256d t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16;
	t1 = _mm256_load_pd(s1);
	t5 = _mm256_load_pd(s1+4);
	t2 = _mm256_load_pd(s2);
	t6 = _mm256_load_pd(s2+4);
	t3 = _mm256_load_pd(s3);
	t7 = _mm256_load_pd(s3+4);
	t4 = _mm256_load_pd(s4);
	t8 = _mm256_load_pd(s4+4);
	t9 = _mm256_load_pd(s5);
	t13 = _mm256_load_pd(s5+4);
	t10 = _mm256_load_pd(s6);
	t14 = _mm256_load_pd(s6+4);
	t11 = _mm256_load_pd(s7);
	t15 = _mm256_load_pd(s7+4);
	t12 = _mm256_load_pd(s8);
	t16 = _mm256_load_pd(s8+4);

	trans_avx256_kernel4x4(t1,t2,t3,t4,d1,d2,d3,d4);
	trans_avx256_kernel4x4(t9,t10,t11,t12,d1+4,d2+4,d3+4,d4+4);
	trans_avx256_kernel4x4(t5,t6,t7,t8,d5,d6,d7,d8);
	trans_avx256_kernel4x4(t13,t14,t15,t16,d5+4,d6+4,d7+4,d8+4);

}

void trans8x8AVX256(double *x, double *y, size_t n)
{
	
	double *s1=x, *s2=x+n, *s3=x+2*n, *s4=x+3*n,
	       *s5=x+4*n, *s6=x+5*n, *s7=x+6*n, *s8=x+7*n;
	double *d1=y, *d2=y+n, *d3=y+2*n, *d4=y+3*n,
		   *d5=y+4*n, *d6=y+5*n, *d7=y+6*n, *d8=y+7*n;

	for(int i=0; i<n; i+=8){
		d1 = y+i;
		d2 = d1+n;
		d3 = d1+2*n;
		d4 = d1+3*n;
		d5 = d1+4*n;
		d6 = d1+5*n;
		d7 = d1+6*n;
		d8 = d1+7*n;
		int j=n;
		while(j>=8){
			trans_avx256_kernel8x8(s1,s2,s3,s4,s5,s6,s7,s8,d1,d2,d3,d4,d5,d6,d7,d8);
			s1+=8;
            s2+=8;
            s3+=8;
            s4+=8;
            s5+=8;
            s6+=8;
            s7+=8;
            s8+=8;
			d1+=8*n;
			d2+=8*n;
			d3+=8*n;
			d4+=8*n;
			d5+=8*n;
			d6+=8*n;
			d7+=8*n;
			d8+=8*n;
			j-=8;
		}
		s1+=7*n;
		s2+=7*n;
		s3+=7*n;
		s4+=7*n;
		s5+=7*n;
		s6+=7*n;
		s7+=7*n;
		s8+=7*n;
	}


}


void trans4x4AVX256(double *x, double *y, size_t n)
{
	double *s1=x, *s2=x+n, *s3=x+2*n, *s4=x+3*n;
	double *d1=y, *d2=y+n, *d3=y+2*n, *d4=y+3*n;
	for(int i=0; i<n; i+=4){
		d1 = y+i;
		d2 = d1+n;
		d3 = d1+2*n;
		d4 = d1+3*n;
		int j = n;
		while(j>=4){
			__m256d vs1,vs2,vs3,vs4;		
			vs1 = _mm256_load_pd(s1);
			vs2 = _mm256_load_pd(s2);
			vs3 = _mm256_load_pd(s3);
			vs4 = _mm256_load_pd(s4);
			trans_avx256_kernel4x4(vs1, vs2, vs3, vs4, d1, d2, d3, d4);
			s1+=4;
			s2+=4;
			s3+=4;
			s4+=4;
			d1+=4*n;
			d2+=4*n;
			d3+=4*n;
			d4+=4*n;
			j-=4;
		}
		s1+=3*n;
		s2+=3*n;
		s3+=3*n;
		s4+=3*n;

	}

}


int main()
{
	double *x=(double *)_mm_malloc(sizeof(double)*N*N, 32);
	double *y=(double *)_mm_malloc(sizeof(double)*N*N, 32);
	double dt;	
	for(int i=0; i<N*N; i++){
		x[i] = i%100;
	}

	clock_t start=clock();
	trans(x,y,N);
	dt=static_cast<double>(clock()-start)/CLOCKS_PER_SEC;

	//test
	cout<<x[J*N+I]<<'\t'<<y[I*N+J]<<'\n';
	cout << "(normal)矩阵转置耗时: " << dt << " s" << endl;

	//AVX256 4x4
	start=clock();
	trans4x4AVX256(x,y,N);
	dt=static_cast<double>(clock()-start)/CLOCKS_PER_SEC;
	
	cout<<x[J*N+I]<<'\t'<<y[I*N+J]<<'\n';
	cout << "(AVX256)4x4 block 矩阵转置耗时: " << dt << " s" << endl;
	
	//AVX256 8x8
	start=clock();
	trans8x8AVX256(x,y,N);
	dt=static_cast<double>(clock()-start)/CLOCKS_PER_SEC;
	
	cout<<x[J*N+I]<<'\t'<<y[I*N+J]<<'\n';
	cout << "(AVX256)8x8 block 矩阵转置耗时: " << dt << " s" << endl;
}

//https://blog.csdn.net/artorias123/article/details/90513600

