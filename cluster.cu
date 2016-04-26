#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <cstring>


int lenString=594;
int maxNumStrings = 1000000;                       
int threshold = 2;

__global__ void Compare(int position, char *d_a, int *d_b, int *d_c, int max_count, int lenString, int threshold){
	int my_id = blockDim.x * blockIdx.x + threadIdx.x;
	if ((my_id < max_count) && (d_c[my_id] == 0) && (my_id != position)){
		int offset = my_id*lenString - position*lenString;
		int x, i, diffs = 0, stop =0;
	for (x=0;x<lenString;x+=6){
		for (i=0;i<6;i++){
			diffs += (bool)(d_a[x+i+position]^d_a[x+i+offset+position]);
			if (diffs > threshold){
				stop += 1;
				break;}
		}
	if (stop == 1)
		break;
	}
	
	if (diffs == threshold){
		d_b[position] += d_b[my_id];
		d_c[position] = 2;
		d_c[my_id] = 1;
	}
}
}

int main(int argc, char** argv) {
	
//allocation of variables
	char *strings;	//host copy of a
	char *d_a;//device copy of a
	int *counts, *d_b;
	int *merged, *d_c;
	char copy[lenString+1]; //string to copy in info
	int i=0, k=0, num;
	int size_string = maxNumStrings*sizeof(char)*(lenString+1);
	int size_int = maxNumStrings*sizeof(int);
	struct timeval start, end; 				//using time
	double wallTime;
	cudaError_t status = (cudaError_t)0;



	//opening the file
	FILE *fp;
	if ((fp=fopen("/cluster/home/charliep/courses/cs360/single-linkage-clustering/Iceland2014.trim.contigs.good.unique.good.filter.unique.count.fasta", "r")) == NULL) 
	{perror("could not open the input file: ");
	exit(1);
	};
	
	merged = (int *)malloc(size_int);		
	
	if (!(strings= (char *)malloc(size_string))) {
		fprintf(stderr, "malloc() FAILED (Block)\n"); 
		exit(0);}
	if (!(counts= (int *)malloc(size_int))) {
		fprintf(stderr, "malloc() FAILED (Block)\n"); 
		exit(0);}


	while( fscanf(fp,"%s %d", copy, &num) != EOF){
		strncpy(&strings[i],copy,lenString);
		counts[k] = num;
		merged[k] = 0;
		//printf("%s\n", copy);
		//printf("%s\n", &a[i]);
		i=i+lenString;
		k++;
		}
	fclose(fp);
	cudaMalloc(&d_a, size_string);
	cudaMalloc(&d_b, size_int);
	cudaMalloc(&d_c, size_int);

	cudaMemcpy(d_a, strings, size_string, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, counts, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, merged, size_int, cudaMemcpyHostToDevice);

	int threads_num = 512, blocks_num;
	blocks_num = (int)ceil((float)k/threads_num);
	
	int position = 0;
	Compare<<<blocks_num, threads_num>>>(position, d_a, d_b, d_c, k, lenString, threshold);
	cudaMemcpy(merged, d_c, size_int, cudaMemcpyDeviceToHost);
	

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	free(strings);
	free(counts);
	free(merged);
}
