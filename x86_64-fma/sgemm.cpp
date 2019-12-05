#include <iostream>
#include <cstdlib>
#include <immintrin.h>

//#include <omp.h>
using namespace std;
typedef double MATRIXTYPE;

#define M 256
#define K 256
#define N 256

class Matrix{
    private:
    	MATRIXTYPE **_Matrix;
   		size_t _Row,_Column;
    public:
    	Matrix():_Matrix(nullptr),_Row(0),_Column(0){}//默认构造
    	Matrix(size_t r,size_t c):_Row(r),_Column(c){//构造r行、c列的矩阵
        	if(!_Column||!_Row) return;
        	_Matrix=(MATRIXTYPE**)malloc(_Column*sizeof(MATRIXTYPE*));
        	MATRIXTYPE **p=_Matrix,**end=_Matrix+_Column;
       		do{
            	*(p++)=(MATRIXTYPE*)malloc(_Row*sizeof(MATRIXTYPE));
        	}while(p!=end);
    	}
    	Matrix(size_t r,size_t c,const MATRIXTYPE init):_Row(r),_Column(c){//构造r行、c列的矩阵并用init初始化
        	if(!_Column||!_Row) return;
        	_Matrix=(MATRIXTYPE**)malloc(_Column*sizeof(MATRIXTYPE*));
        	MATRIXTYPE **pr=_Matrix,**endr=_Matrix+_Column,*p,*end;
        	do{
            	p=*(pr++)=(MATRIXTYPE*)malloc(_Row*sizeof(MATRIXTYPE));
            	end=p+_Row;
            	do{
                	*(p++)=init;
            	}while(p!=end);
        	}while(pr!=endr);
    	}
    	Matrix(const Matrix& B){//拷贝构造
        	_Row=B._Row;
        	_Column=B._Column;
        	_Matrix=(MATRIXTYPE**)malloc(_Column*sizeof(MATRIXTYPE*));
        	MATRIXTYPE **pbr=B._Matrix,**endbr=B._Matrix+_Column,*pb,*endb,
                       **par=_Matrix,**endar=_Matrix+_Column,*pa,*enda;
        	do{
            	pa=*(par++)=(MATRIXTYPE*)malloc(_Row*sizeof(MATRIXTYPE));
            	enda=pa+_Row;
            	pb=*(pbr++);
            	endb=pb+_Row;
            	do{
                	*(pa++)=*(pb++);
            	}while(pa!=enda);
        	}while(par!=endar);
    	}

    	~Matrix(){//析构
        	if(!_Matrix) return;
        	MATRIXTYPE **p=_Matrix,**end=_Matrix+_Column;
        	do{
            	free(*(p++));
        	}while(p!=end);
        	_Column=_Row=0;
        	free(_Matrix);
    	}

    	MATRIXTYPE& operator()(size_t i,size_t j){return _Matrix[j][i];}//访问第i行、第j列的元素
    
		const MATRIXTYPE operator()(size_t i,size_t j)const{return _Matrix[j][i];}//访问第i行、第j列的元素
    
		Matrix& operator=(Matrix&& B){//移动赋值
        	if(_Matrix){
            	MATRIXTYPE **p=_Matrix,**end=_Matrix+_Row;
            	do{
                	free(*(p++));
            	}while(p!=end);
            	free(_Matrix);
        	}
        	_Row=B._Row;
        	_Column=B._Column;
        	_Matrix=B._Matrix;
        	B._Matrix=nullptr;
        	return *this;
    	}

		//按列计算
		Matrix multi1(const Matrix &B){
			if(_Column != B._Row) return *this;
			Matrix tmp(_Row, B._Column, 0);
			int i,j(0),k;
			do{
				i = 0;
				do{
					k = 0;
					do{
						tmp(i,j) += (*this)(i,k)*B(k,j);
						k++;
					}while(k<_Column);
					i++;
				}while(i<_Row);
				j++;
			}while(j<B._Column);
			return tmp;
		}	

		//按行计算
		Matrix multi2(const Matrix &B){
     	   if(_Column!=B._Row) return *this;
        	Matrix tmp(_Row,B._Column,0);
        	int i(0),j,k;
        	do{
            	j=0;
            	do{
                	k=0;
                	do{
                    	tmp(i,j)+=(*this)(i,k)*B(k,j);
                    	k++;
                	}while(k<_Column);
                	j++;
            	}while(j<B._Column);
            	i++;
        	}while(i<_Row);
        	return tmp;
    	}

		//指针代替()重载
		Matrix multi3(const Matrix &B){
        	if(_Column!=B._Row) return *this;
        	Matrix tmp(_Row,B._Column,0);
        	int i(0),j(0),k;
        	MATRIXTYPE **pr,*p;
        	do{
            	j=0;
            	pr=B._Matrix;
            	do{
                	k=0;p=*(pr++);
                	do{
                   		tmp(i,j)+=(*this)(i,k)**(p++);
                    	k++;
                	}while(k<_Column);
                	j++;
            	}while(j<B._Column);
            	i++;
        	}while(i<_Row);
        	return tmp;
    	}


		void multi4kernel(MATRIXTYPE **c,MATRIXTYPE **a,MATRIXTYPE **b,int row,int col){
        	register MATRIXTYPE t0(0),t1(0),t2(0),t3(0),t4(0),t5(0),t6(0),t7(0),
            	            t8(0),t9(0),t10(0),t11(0),t12(0),t13(0),t14(0),t15(0);
        	MATRIXTYPE *a0(a[0]),*a1(a[1]),*a2(a[2]),*a3(a[3]),
            	   *b0(b[col]),*b1(b[col+1]),*b2(b[col+2]),*b3(b[col+3]),*end=b0+_Row;
        	do{
            	t0+=*(a0)**(b0);
           		t1+=*(a0)**(b1);
            	t2+=*(a0)**(b2);
            	t3+=*(a0++)**(b3);
            	t4+=*(a1)**(b0);
            	t5+=*(a1)**(b1);
            	t6+=*(a1)**(b2);
            	t7+=*(a1++)**(b3);
            	t8+=*(a2)**(b0);
            	t9+=*(a2)**(b1);
            	t10+=*(a2)**(b2);
            	t11+=*(a2++)**(b3);
            	t12+=*(a3)**(b0++);
            	t13+=*(a3)**(b1++);
            	t14+=*(a3)**(b2++);
            	t15+=*(a3++)**(b3++);
        	}while(b0!=end);
        	c[col][row]=t0;
        	c[col+1][row]=t1;
        	c[col+2][row]=t2;
        	c[col+3][row]=t3;
        	c[col][row+1]=t4;
        	c[col+1][row+1]=t5;
        	c[col+2][row+1]=t6;
        	c[col+3][row+1]=t7;
        	c[col][row+2]=t8;
        	c[col+1][row+2]=t9;
       		c[col+2][row+2]=t10;
        	c[col+3][row+2]=t11;
        	c[col][row+3]=t12;
        	c[col+1][row+3]=t13;
        	c[col+2][row+3]=t14;
        	c[col+3][row+3]=t15;
    	}

		//pack 4
		Matrix multi4(const Matrix &B){
        	if(_Column!=B._Row) return *this;
        	Matrix tmp(_Row,B._Column,0);
        	MATRIXTYPE *tr[4];
        	int i(0),j(0);
        	do{
            	tr[i++]=(MATRIXTYPE*)malloc(sizeof(MATRIXTYPE)*_Column);
        	}while(i<4);
        	do{
            	i=0;
            	do{
                	tr[0][i]=_Matrix[i][j];//packing过程，把行数据打包到连续空间
                	tr[1][i]=_Matrix[i][j+1];
                	tr[2][i]=_Matrix[i][j+2];
                	tr[3][i]=_Matrix[i][j+3];
            	}while((++i)<_Column);
            	i=0;
            	do{
                	multi4kernel(tmp._Matrix,tr,B._Matrix,j,i);
                	i+=4;
            	}while(i<B._Column);
            	j+=4;
        	}while(j<_Row);
        	return tmp;
    	}

		//SIMD,利用SSE指令加速
		void multi5kernel(MATRIXTYPE **c,MATRIXTYPE **a,MATRIXTYPE **b,int row,int col){
        	__m128d t01_0,t01_1,t01_2,t01_3,t23_0,t23_1,t23_2,t23_3,
                	a0,a1,b0,b1,b2,b3;
        	t01_0=t01_1=t01_2=t01_3=t23_0=t23_1=t23_2=t23_3=_mm_set1_pd(0);
        	MATRIXTYPE *pb0(b[col]),*pb1(b[col+1]),*pb2(b[col+2]),*pb3(b[col+3]),*pa0(a[0]),*pa1(a[1]),*endb0=pb0+_Column;
        	do{
            	a0=_mm_load_pd(pa0);
            	a1=_mm_load_pd(pa1);
            	b0=_mm_set1_pd(*(pb0++));
            	b1=_mm_set1_pd(*(pb1++));
            	b2=_mm_set1_pd(*(pb2++));
            	b3=_mm_set1_pd(*(pb3++));
            	t01_0+=a0*b0;
            	t01_1+=a0*b1;
            	t01_2+=a0*b2;
            	t01_3+=a0*b3;
            	t23_0+=a1*b0;
            	t23_1+=a1*b1;
            	t23_2+=a1*b2;
            	t23_3+=a1*b3;
            	pa0+=2;
            	pa1+=2;
        	}while(pb0!=endb0);
        	_mm_store_pd(&c[col][row],t01_0);
        	_mm_store_pd(&c[col+1][row],t01_1);
        	_mm_store_pd(&c[col+2][row],t01_2);
        	_mm_store_pd(&c[col+3][row],t01_3);
        	_mm_store_pd(&c[col][row+2],t23_0);
        	_mm_store_pd(&c[col+1][row+2],t23_1);
        	_mm_store_pd(&c[col+2][row+2],t23_2);
        	_mm_store_pd(&c[col+3][row+2],t23_3);
    	}

    	Matrix multi5(const Matrix &B){
        	if(_Column!=B._Row) return *this;
        	Matrix tmp(_Row,B._Column,0);
        	MATRIXTYPE *ta[2];
        	//MATRIXTYPE *tb;
        	ta[0]=(MATRIXTYPE*)malloc(sizeof(MATRIXTYPE)*2*_Column);
        	ta[1]=(MATRIXTYPE*)malloc(sizeof(MATRIXTYPE)*2*_Column);
        	//tb=(MATRIXTYPE*)malloc(sizeof(MATRIXTYPE)*4*B._Row);
        	//end=tb+4*B._Row;
        	//end=tb;
        	int i(0),j(0),k,t;
        	do{
            	k=0;i=0;
            	do{
                	ta[0][k]=_Matrix[i][j];
                	ta[1][k++]=_Matrix[i][j+2];
                	ta[0][k]=_Matrix[i][j+1];
                	ta[1][k++]=_Matrix[i++][j+3];
            	}while(i<_Column);
            	i=0;
            	do{
                	multi5kernel(tmp._Matrix,ta,B._Matrix,j,i);
                	i+=4;
            	}while(i<B._Column);
            	j+=4;
        	}while(j<_Row);
        	//free(tb);
        	free(ta[0]);
        	free(ta[1]);
        	return tmp;
    	}

		//利用AVX256加速
		void multi6kernel(MATRIXTYPE **c,MATRIXTYPE **a,MATRIXTYPE **b,int row,int col){
        	__m256d t04_0,t04_1,t04_2,t04_3,t58_0,t58_1,t58_2,t58_3,
                	a0,a1,b0,b1,b2,b3;
        	t04_0=t04_1=t04_2=t04_3=t58_0=t58_1=t58_2=t58_3=_mm256_set1_pd(0);
        	MATRIXTYPE *pb0(b[col]),*pb1(b[col+1]),*pb2(b[col+2]),*pb3(b[col+3]),*pa0(a[0]),*pa1(a[1]),*endb0=pb0+_Column;
        	do{
            	a0=_mm256_loadu_pd(pa0);
            	a1=_mm256_loadu_pd(pa1);
            	b0=_mm256_set1_pd(*(pb0++));
            	b1=_mm256_set1_pd(*(pb1++));
            	b2=_mm256_set1_pd(*(pb2++));
            	b3=_mm256_set1_pd(*(pb3++));
            	t04_0+=a0*b0;
            	t04_1+=a0*b1;
           		t04_2+=a0*b2;
            	t04_3+=a0*b3;
            	t58_0+=a1*b0;
            	t58_1+=a1*b1;
            	t58_2+=a1*b2;
            	t58_3+=a1*b3;
            	pa0+=4;
            	pa1+=4;
        	}while(pb0!=endb0);
        	_mm256_storeu_pd(&c[col][row],t04_0);
        	_mm256_storeu_pd(&c[col+1][row],t04_1);
        	_mm256_storeu_pd(&c[col+2][row],t04_2);
        	_mm256_storeu_pd(&c[col+3][row],t04_3);
        	_mm256_storeu_pd(&c[col][row+4],t58_0);
        	_mm256_storeu_pd(&c[col+1][row+4],t58_1);
        	_mm256_storeu_pd(&c[col+2][row+4],t58_2);
        	_mm256_storeu_pd(&c[col+3][row+4],t58_3);
    	}

    	Matrix multi6(const Matrix &B){
        	if(_Column!=B._Row) return *this;
        	Matrix tmp(_Row,B._Column,0);
        	MATRIXTYPE *ta[2];
        	ta[0]=(MATRIXTYPE*)malloc(sizeof(MATRIXTYPE)*4*_Column);
        	ta[1]=(MATRIXTYPE*)malloc(sizeof(MATRIXTYPE)*4*_Column);
        	int i(0),j(0),k,t;
        	do{
            	k=0;i=0;
            	do{
                	ta[0][k]=_Matrix[i][j];
                	ta[1][k++]=_Matrix[i][j+4];
                	ta[0][k]=_Matrix[i][j+1];
                	ta[1][k++]=_Matrix[i][j+5];
                	ta[0][k]=_Matrix[i][j+2];
                	ta[1][k++]=_Matrix[i][j+6];
                	ta[0][k]=_Matrix[i][j+3];
                	ta[1][k++]=_Matrix[i++][j+7];
            	}while(i<_Column);
            	i=0;
            	do{
                	multi6kernel(tmp._Matrix,ta,B._Matrix,j,i);
                	i+=4;
            	}while(i<B._Column);
            	j+=8;
        	}while(j<_Row);
        	free(ta[0]);
        	free(ta[1]);
        	return tmp;
   		}

};


int main()
{
	Matrix A(M, K, 1), B(K, N, 2);
	//cout << A(1,1) << endl;
	//cout << B(2,2) << endl;
	MATRIXTYPE dt;
	int loops = 200;
    clock_t start=clock();
	//#pragma omp parallel for num_threads(12)
	for(int i=0; i<loops; i++){
    	A.multi1(B);
	}
    dt=static_cast<double>(clock()-start)/CLOCKS_PER_SEC;
    cout<<"按列计算 multi1 用时：\n"<<dt<<"s"<<endl;


    start=clock();
	for(int i=0; i<loops; i++){
    	A.multi2(B);
	}
    dt=static_cast<double>(clock()-start)/CLOCKS_PER_SEC;
    cout<<"按行计算 multi2 用时：\n"<<dt<<"s"<<endl;

    start=clock();
	for(int i=0; i<loops; i++){
    	A.multi3(B);
	}
    dt=static_cast<double>(clock()-start)/CLOCKS_PER_SEC;
    cout<<"指针代替()重载 multi3 用时：\n"<<dt<<"s"<<endl;

    start=clock();
	for(int i=0; i<loops; i++){
    	A.multi4(B);
	}
    dt=static_cast<double>(clock()-start)/CLOCKS_PER_SEC;
    cout<<"pack4 multi4 用时：\n"<<dt<<"s"<<endl;

    start=clock();
	for(int i=0; i<loops; i++){
    	A.multi5(B);
	}
    dt=static_cast<double>(clock()-start)/CLOCKS_PER_SEC;
    cout<<"SSE 指令 multi5 用时：\n"<<dt<<"s"<<endl;

    start=clock();
	for(int i=0; i<loops; i++){
    	A.multi6(B);
	}
    dt=static_cast<double>(clock()-start)/CLOCKS_PER_SEC;
    cout<<"AVX256 指令 multi6 用时：\n"<<dt<<"s"<<endl;

	return 0;
}
