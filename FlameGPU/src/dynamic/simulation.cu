
/*
 * FLAME GPU v 1.4.0 for CUDA 6
 * Copyright 2015 University of Sheffield.
 * Author: Dr Paul Richmond 
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence 
 * on www.flamegpu.com website.
 * 
 */

//Disable internal thrust warnings about conversions
#pragma warning(push)
#pragma warning (disable : 4267)
#pragma warning (disable : 4244)

// includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

// include FLAME kernels
#include "FLAMEGPU_kernals.cu"


#pragma warning(pop)

/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Error check function for post CUDA Kernel calling */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
inline void gpuLaunchAssert(const char *file, int line, bool abort=true)
{
	gpuAssert( cudaPeekAtLastError(), file, line );
#ifdef _DEBUG
	gpuAssert( cudaDeviceSynchronize(), file, line );
#endif
   
}

/* SM padding and offset variables */
int SM_START;
int PADDING;

/* Agent Memory */

/* LTin Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_LTin_list* d_LTins;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_LTin_list* d_LTins_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_LTin_list* d_LTins_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_LTin_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_LTin_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_LTin_values;  /**< Agent sort identifiers value */
    
/* LTin state variables */
xmachine_memory_LTin_list* h_LTins_ltin_random_movement;      /**< Pointer to agent list (population) on host*/
xmachine_memory_LTin_list* d_LTins_ltin_random_movement;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_LTin_ltin_random_movement_count;   /**< Agent population size counter */ 

/* LTin state variables */
xmachine_memory_LTin_list* h_LTins_stable_contact;      /**< Pointer to agent list (population) on host*/
xmachine_memory_LTin_list* d_LTins_stable_contact;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_LTin_stable_contact_count;   /**< Agent population size counter */ 

/* LTin state variables */
xmachine_memory_LTin_list* h_LTins_localised_movement;      /**< Pointer to agent list (population) on host*/
xmachine_memory_LTin_list* d_LTins_localised_movement;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_LTin_localised_movement_count;   /**< Agent population size counter */ 

/* LTi Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_LTi_list* d_LTis;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_LTi_list* d_LTis_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_LTi_list* d_LTis_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_LTi_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_LTi_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_LTi_values;  /**< Agent sort identifiers value */
    
/* LTi state variables */
xmachine_memory_LTi_list* h_LTis_lti_random_movement;      /**< Pointer to agent list (population) on host*/
xmachine_memory_LTi_list* d_LTis_lti_random_movement;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_LTi_lti_random_movement_count;   /**< Agent population size counter */ 

/* LTi state variables */
xmachine_memory_LTi_list* h_LTis_responding;      /**< Pointer to agent list (population) on host*/
xmachine_memory_LTi_list* d_LTis_responding;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_LTi_responding_count;   /**< Agent population size counter */ 

/* LTi state variables */
xmachine_memory_LTi_list* h_LTis_contact;      /**< Pointer to agent list (population) on host*/
xmachine_memory_LTi_list* d_LTis_contact;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_LTi_contact_count;   /**< Agent population size counter */ 

/* LTi state variables */
xmachine_memory_LTi_list* h_LTis_adhesion;      /**< Pointer to agent list (population) on host*/
xmachine_memory_LTi_list* d_LTis_adhesion;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_LTi_adhesion_count;   /**< Agent population size counter */ 

/* LTo Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_LTo_list* d_LTos;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_LTo_list* d_LTos_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_LTo_list* d_LTos_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_LTo_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_LTo_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_LTo_values;  /**< Agent sort identifiers value */
    
/* LTo state variables */
xmachine_memory_LTo_list* h_LTos_no_expression;      /**< Pointer to agent list (population) on host*/
xmachine_memory_LTo_list* d_LTos_no_expression;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_LTo_no_expression_count;   /**< Agent population size counter */ 

/* LTo state variables */
xmachine_memory_LTo_list* h_LTos_expression;      /**< Pointer to agent list (population) on host*/
xmachine_memory_LTo_list* d_LTos_expression;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_LTo_expression_count;   /**< Agent population size counter */ 

/* LTo state variables */
xmachine_memory_LTo_list* h_LTos_adhesion_upregulation;      /**< Pointer to agent list (population) on host*/
xmachine_memory_LTo_list* d_LTos_adhesion_upregulation;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_LTo_adhesion_upregulation_count;   /**< Agent population size counter */ 

/* LTo state variables */
xmachine_memory_LTo_list* h_LTos_chemokine_upregulation;      /**< Pointer to agent list (population) on host*/
xmachine_memory_LTo_list* d_LTos_chemokine_upregulation;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_LTo_chemokine_upregulation_count;   /**< Agent population size counter */ 

/* LTo state variables */
xmachine_memory_LTo_list* h_LTos_mature;      /**< Pointer to agent list (population) on host*/
xmachine_memory_LTo_list* d_LTos_mature;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_LTo_mature_count;   /**< Agent population size counter */ 

/* LTo state variables */
xmachine_memory_LTo_list* h_LTos_downregulated;      /**< Pointer to agent list (population) on host*/
xmachine_memory_LTo_list* d_LTos_downregulated;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_LTo_downregulated_count;   /**< Agent population size counter */ 


/* Message Memory */

/* location Message variables */
xmachine_message_location_list* h_locations;         /**< Pointer to message list on host*/
xmachine_message_location_list* d_locations;         /**< Pointer to message list on device*/
xmachine_message_location_list* d_locations_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_location_count;         /**< message list counter*/
int h_message_location_output_type;   /**< message output type (single or optional)*/

  
/* CUDA Streams for function layers */
cudaStream_t stream1;
cudaStream_t stream2;

/*Global condition counts*/

/* RNG rand48 */
RNG_rand48* h_rand48;    /**< Pointer to RNG_rand48 seed list on host*/
RNG_rand48* d_rand48;    /**< Pointer to RNG_rand48 seed list on device*/

/* Cuda Event Timers for Instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEvent_t instrument_iteration_start, instrument_iteration_stop;
	float instrument_iteration_milliseconds = 0.0f;
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEvent_t instrument_start, instrument_stop;
	float instrument_milliseconds = 0.0f;
#endif

/* CUDA Parallel Primatives variables */
int scan_last_sum;           /**< Indicates if the position (in message list) of last message*/
int scan_last_included;      /**< Indicates if last sum value is included in the total sum count*/

/* Agent function prototypes */

/** LTin_ltin_random_move
 * Agent function prototype for ltin_random_move function of LTin agent
 */
void LTin_ltin_random_move(cudaStream_t &stream);

/** LTi_lti_random_move
 * Agent function prototype for lti_random_move function of LTi agent
 */
void LTi_lti_random_move(cudaStream_t &stream);

  
void setPaddingAndOffset()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int x64_sys = 0;

	// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
	if (deviceProp.major == 9999 && deviceProp.minor == 9999){
		printf("Error: There is no device supporting CUDA.\n");
		exit(0);
	}
    
    //check if double is used and supported
#ifdef _DOUBLE_SUPPORT_REQUIRED_
	printf("Simulation requires full precision double values\n");
	if ((deviceProp.major < 2)&&(deviceProp.minor < 3)){
		printf("Error: Hardware does not support full precision double values!\n");
		exit(0);
	}
    
#endif

	//check 32 or 64bit
	x64_sys = (sizeof(void*)==8);
	if (x64_sys)
	{
		printf("64Bit System Detected\n");
	}
	else
	{
		printf("32Bit System Detected\n");
	}

	SM_START = 0;
	PADDING = 0;
  
	//copy padding and offset to GPU
	gpuErrchk(cudaMemcpyToSymbol( d_SM_START, &SM_START, sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol( d_PADDING, &PADDING, sizeof(int)));     
}

int is_sqr_pow2(int x){
	int r = (int)pow(4, ceil(log(x)/log(4)));
	return (r == x);
}

int lowest_sqr_pow2(int x){
	int l;
	
	//escape early if x is square power of 2
	if (is_sqr_pow2(x))
		return x;
	
	//lower bound		
	l = (int)pow(4, floor(log(x)/log(4)));
	
	return l;
}

/* Unary function required for cudaOccupancyMaxPotentialBlockSizeVariableSMem to avoid warnings */
int no_sm(int b){
	return 0;
}

/* Unary function to return shared memory size for reorder message kernels */
int reorder_messages_sm_size(int blockSize)
{
	return sizeof(unsigned int)*(blockSize+1);
}


void initialise(char * inputfile){

	//set the padding and offset values depending on architecture and OS
	setPaddingAndOffset();
  

	printf("Allocating Host and Device memory\n");
  
	/* Agent memory allocation (CPU) */
	int xmachine_LTin_SoA_size = sizeof(xmachine_memory_LTin_list);
	h_LTins_ltin_random_movement = (xmachine_memory_LTin_list*)malloc(xmachine_LTin_SoA_size);
	h_LTins_stable_contact = (xmachine_memory_LTin_list*)malloc(xmachine_LTin_SoA_size);
	h_LTins_localised_movement = (xmachine_memory_LTin_list*)malloc(xmachine_LTin_SoA_size);
	int xmachine_LTi_SoA_size = sizeof(xmachine_memory_LTi_list);
	h_LTis_lti_random_movement = (xmachine_memory_LTi_list*)malloc(xmachine_LTi_SoA_size);
	h_LTis_responding = (xmachine_memory_LTi_list*)malloc(xmachine_LTi_SoA_size);
	h_LTis_contact = (xmachine_memory_LTi_list*)malloc(xmachine_LTi_SoA_size);
	h_LTis_adhesion = (xmachine_memory_LTi_list*)malloc(xmachine_LTi_SoA_size);
	int xmachine_LTo_SoA_size = sizeof(xmachine_memory_LTo_list);
	h_LTos_no_expression = (xmachine_memory_LTo_list*)malloc(xmachine_LTo_SoA_size);
	h_LTos_expression = (xmachine_memory_LTo_list*)malloc(xmachine_LTo_SoA_size);
	h_LTos_adhesion_upregulation = (xmachine_memory_LTo_list*)malloc(xmachine_LTo_SoA_size);
	h_LTos_chemokine_upregulation = (xmachine_memory_LTo_list*)malloc(xmachine_LTo_SoA_size);
	h_LTos_mature = (xmachine_memory_LTo_list*)malloc(xmachine_LTo_SoA_size);
	h_LTos_downregulated = (xmachine_memory_LTo_list*)malloc(xmachine_LTo_SoA_size);

	/* Message memory allocation (CPU) */
	int message_location_SoA_size = sizeof(xmachine_message_location_list);
	h_locations = (xmachine_message_location_list*)malloc(message_location_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outputs

	//read initial states
	readInitialStates(inputfile, h_LTins_ltin_random_movement, &h_xmachine_memory_LTin_ltin_random_movement_count, h_LTis_lti_random_movement, &h_xmachine_memory_LTi_lti_random_movement_count, h_LTos_no_expression, &h_xmachine_memory_LTo_no_expression_count);
	
	
	/* LTin Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_LTins, xmachine_LTin_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_LTins_swap, xmachine_LTin_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_LTins_new, xmachine_LTin_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_LTin_keys, xmachine_memory_LTin_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_LTin_values, xmachine_memory_LTin_MAX* sizeof(uint)));
	/* ltin_random_movement memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_LTins_ltin_random_movement, xmachine_LTin_SoA_size));
	gpuErrchk( cudaMemcpy( d_LTins_ltin_random_movement, h_LTins_ltin_random_movement, xmachine_LTin_SoA_size, cudaMemcpyHostToDevice));
    
	/* stable_contact memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_LTins_stable_contact, xmachine_LTin_SoA_size));
	gpuErrchk( cudaMemcpy( d_LTins_stable_contact, h_LTins_stable_contact, xmachine_LTin_SoA_size, cudaMemcpyHostToDevice));
    
	/* localised_movement memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_LTins_localised_movement, xmachine_LTin_SoA_size));
	gpuErrchk( cudaMemcpy( d_LTins_localised_movement, h_LTins_localised_movement, xmachine_LTin_SoA_size, cudaMemcpyHostToDevice));
    
	/* LTi Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_LTis, xmachine_LTi_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_LTis_swap, xmachine_LTi_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_LTis_new, xmachine_LTi_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_LTi_keys, xmachine_memory_LTi_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_LTi_values, xmachine_memory_LTi_MAX* sizeof(uint)));
	/* lti_random_movement memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_LTis_lti_random_movement, xmachine_LTi_SoA_size));
	gpuErrchk( cudaMemcpy( d_LTis_lti_random_movement, h_LTis_lti_random_movement, xmachine_LTi_SoA_size, cudaMemcpyHostToDevice));
    
	/* responding memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_LTis_responding, xmachine_LTi_SoA_size));
	gpuErrchk( cudaMemcpy( d_LTis_responding, h_LTis_responding, xmachine_LTi_SoA_size, cudaMemcpyHostToDevice));
    
	/* contact memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_LTis_contact, xmachine_LTi_SoA_size));
	gpuErrchk( cudaMemcpy( d_LTis_contact, h_LTis_contact, xmachine_LTi_SoA_size, cudaMemcpyHostToDevice));
    
	/* adhesion memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_LTis_adhesion, xmachine_LTi_SoA_size));
	gpuErrchk( cudaMemcpy( d_LTis_adhesion, h_LTis_adhesion, xmachine_LTi_SoA_size, cudaMemcpyHostToDevice));
    
	/* LTo Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_LTos, xmachine_LTo_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_LTos_swap, xmachine_LTo_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_LTos_new, xmachine_LTo_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_LTo_keys, xmachine_memory_LTo_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_LTo_values, xmachine_memory_LTo_MAX* sizeof(uint)));
	/* no_expression memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_LTos_no_expression, xmachine_LTo_SoA_size));
	gpuErrchk( cudaMemcpy( d_LTos_no_expression, h_LTos_no_expression, xmachine_LTo_SoA_size, cudaMemcpyHostToDevice));
    
	/* expression memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_LTos_expression, xmachine_LTo_SoA_size));
	gpuErrchk( cudaMemcpy( d_LTos_expression, h_LTos_expression, xmachine_LTo_SoA_size, cudaMemcpyHostToDevice));
    
	/* adhesion_upregulation memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_LTos_adhesion_upregulation, xmachine_LTo_SoA_size));
	gpuErrchk( cudaMemcpy( d_LTos_adhesion_upregulation, h_LTos_adhesion_upregulation, xmachine_LTo_SoA_size, cudaMemcpyHostToDevice));
    
	/* chemokine_upregulation memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_LTos_chemokine_upregulation, xmachine_LTo_SoA_size));
	gpuErrchk( cudaMemcpy( d_LTos_chemokine_upregulation, h_LTos_chemokine_upregulation, xmachine_LTo_SoA_size, cudaMemcpyHostToDevice));
    
	/* mature memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_LTos_mature, xmachine_LTo_SoA_size));
	gpuErrchk( cudaMemcpy( d_LTos_mature, h_LTos_mature, xmachine_LTo_SoA_size, cudaMemcpyHostToDevice));
    
	/* downregulated memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_LTos_downregulated, xmachine_LTo_SoA_size));
	gpuErrchk( cudaMemcpy( d_LTos_downregulated, h_LTos_downregulated, xmachine_LTo_SoA_size, cudaMemcpyHostToDevice));
    
	/* location Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_locations, message_location_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_locations_swap, message_location_SoA_size));
	gpuErrchk( cudaMemcpy( d_locations, h_locations, message_location_SoA_size, cudaMemcpyHostToDevice));
		

	/*Set global condition counts*/

	/* RNG rand48 */
	int h_rand48_SoA_size = sizeof(RNG_rand48);
	h_rand48 = (RNG_rand48*)malloc(h_rand48_SoA_size);
	//allocate on GPU
	gpuErrchk( cudaMalloc( (void**) &d_rand48, h_rand48_SoA_size));
	// calculate strided iteration constants
	static const unsigned long long a = 0x5DEECE66DLL, c = 0xB;
	int seed = 123;
	unsigned long long A, C;
	A = 1LL; C = 0LL;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		C += A*c;
		A *= a;
	}
	h_rand48->A.x = A & 0xFFFFFFLL;
	h_rand48->A.y = (A >> 24) & 0xFFFFFFLL;
	h_rand48->C.x = C & 0xFFFFFFLL;
	h_rand48->C.y = (C >> 24) & 0xFFFFFFLL;
	// prepare first nThreads random numbers from seed
	unsigned long long x = (((unsigned long long)seed) << 16) | 0x330E;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		x = a*x + c;
		h_rand48->seeds[i].x = x & 0xFFFFFFLL;
		h_rand48->seeds[i].y = (x >> 24) & 0xFFFFFFLL;
	}
	//copy to device
	gpuErrchk( cudaMemcpy( d_rand48, h_rand48, h_rand48_SoA_size, cudaMemcpyHostToDevice));

	/* Call all init functions */
	/* Prepare cuda event timers for instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventCreate(&instrument_iteration_start);
	cudaEventCreate(&instrument_iteration_stop);
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEventCreate(&instrument_start);
	cudaEventCreate(&instrument_stop);
#endif

	
  
  /* Init CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamCreate(&stream1));
  gpuErrchk(cudaStreamCreate(&stream2));
} 


void sort_LTins_ltin_random_movement(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTin_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_LTin_ltin_random_movement_count); 
	gridSize = (h_xmachine_memory_LTin_ltin_random_movement_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_LTin_keys, d_xmachine_memory_LTin_values, d_LTins_ltin_random_movement);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_LTin_keys),  thrust::device_pointer_cast(d_xmachine_memory_LTin_keys) + h_xmachine_memory_LTin_ltin_random_movement_count,  thrust::device_pointer_cast(d_xmachine_memory_LTin_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_LTin_agents, no_sm, h_xmachine_memory_LTin_ltin_random_movement_count); 
	gridSize = (h_xmachine_memory_LTin_ltin_random_movement_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_LTin_agents<<<gridSize, blockSize>>>(d_xmachine_memory_LTin_values, d_LTins_ltin_random_movement, d_LTins_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_LTin_list* d_LTins_temp = d_LTins_ltin_random_movement;
	d_LTins_ltin_random_movement = d_LTins_swap;
	d_LTins_swap = d_LTins_temp;	
}

void sort_LTins_stable_contact(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTin_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_LTin_stable_contact_count); 
	gridSize = (h_xmachine_memory_LTin_stable_contact_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_LTin_keys, d_xmachine_memory_LTin_values, d_LTins_stable_contact);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_LTin_keys),  thrust::device_pointer_cast(d_xmachine_memory_LTin_keys) + h_xmachine_memory_LTin_stable_contact_count,  thrust::device_pointer_cast(d_xmachine_memory_LTin_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_LTin_agents, no_sm, h_xmachine_memory_LTin_stable_contact_count); 
	gridSize = (h_xmachine_memory_LTin_stable_contact_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_LTin_agents<<<gridSize, blockSize>>>(d_xmachine_memory_LTin_values, d_LTins_stable_contact, d_LTins_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_LTin_list* d_LTins_temp = d_LTins_stable_contact;
	d_LTins_stable_contact = d_LTins_swap;
	d_LTins_swap = d_LTins_temp;	
}

void sort_LTins_localised_movement(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTin_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_LTin_localised_movement_count); 
	gridSize = (h_xmachine_memory_LTin_localised_movement_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_LTin_keys, d_xmachine_memory_LTin_values, d_LTins_localised_movement);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_LTin_keys),  thrust::device_pointer_cast(d_xmachine_memory_LTin_keys) + h_xmachine_memory_LTin_localised_movement_count,  thrust::device_pointer_cast(d_xmachine_memory_LTin_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_LTin_agents, no_sm, h_xmachine_memory_LTin_localised_movement_count); 
	gridSize = (h_xmachine_memory_LTin_localised_movement_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_LTin_agents<<<gridSize, blockSize>>>(d_xmachine_memory_LTin_values, d_LTins_localised_movement, d_LTins_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_LTin_list* d_LTins_temp = d_LTins_localised_movement;
	d_LTins_localised_movement = d_LTins_swap;
	d_LTins_swap = d_LTins_temp;	
}

void sort_LTis_lti_random_movement(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTi_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_LTi_lti_random_movement_count); 
	gridSize = (h_xmachine_memory_LTi_lti_random_movement_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_LTi_keys, d_xmachine_memory_LTi_values, d_LTis_lti_random_movement);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_LTi_keys),  thrust::device_pointer_cast(d_xmachine_memory_LTi_keys) + h_xmachine_memory_LTi_lti_random_movement_count,  thrust::device_pointer_cast(d_xmachine_memory_LTi_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_LTi_agents, no_sm, h_xmachine_memory_LTi_lti_random_movement_count); 
	gridSize = (h_xmachine_memory_LTi_lti_random_movement_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_LTi_agents<<<gridSize, blockSize>>>(d_xmachine_memory_LTi_values, d_LTis_lti_random_movement, d_LTis_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_LTi_list* d_LTis_temp = d_LTis_lti_random_movement;
	d_LTis_lti_random_movement = d_LTis_swap;
	d_LTis_swap = d_LTis_temp;	
}

void sort_LTis_responding(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTi_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_LTi_responding_count); 
	gridSize = (h_xmachine_memory_LTi_responding_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_LTi_keys, d_xmachine_memory_LTi_values, d_LTis_responding);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_LTi_keys),  thrust::device_pointer_cast(d_xmachine_memory_LTi_keys) + h_xmachine_memory_LTi_responding_count,  thrust::device_pointer_cast(d_xmachine_memory_LTi_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_LTi_agents, no_sm, h_xmachine_memory_LTi_responding_count); 
	gridSize = (h_xmachine_memory_LTi_responding_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_LTi_agents<<<gridSize, blockSize>>>(d_xmachine_memory_LTi_values, d_LTis_responding, d_LTis_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_LTi_list* d_LTis_temp = d_LTis_responding;
	d_LTis_responding = d_LTis_swap;
	d_LTis_swap = d_LTis_temp;	
}

void sort_LTis_contact(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTi_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_LTi_contact_count); 
	gridSize = (h_xmachine_memory_LTi_contact_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_LTi_keys, d_xmachine_memory_LTi_values, d_LTis_contact);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_LTi_keys),  thrust::device_pointer_cast(d_xmachine_memory_LTi_keys) + h_xmachine_memory_LTi_contact_count,  thrust::device_pointer_cast(d_xmachine_memory_LTi_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_LTi_agents, no_sm, h_xmachine_memory_LTi_contact_count); 
	gridSize = (h_xmachine_memory_LTi_contact_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_LTi_agents<<<gridSize, blockSize>>>(d_xmachine_memory_LTi_values, d_LTis_contact, d_LTis_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_LTi_list* d_LTis_temp = d_LTis_contact;
	d_LTis_contact = d_LTis_swap;
	d_LTis_swap = d_LTis_temp;	
}

void sort_LTis_adhesion(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTi_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_LTi_adhesion_count); 
	gridSize = (h_xmachine_memory_LTi_adhesion_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_LTi_keys, d_xmachine_memory_LTi_values, d_LTis_adhesion);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_LTi_keys),  thrust::device_pointer_cast(d_xmachine_memory_LTi_keys) + h_xmachine_memory_LTi_adhesion_count,  thrust::device_pointer_cast(d_xmachine_memory_LTi_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_LTi_agents, no_sm, h_xmachine_memory_LTi_adhesion_count); 
	gridSize = (h_xmachine_memory_LTi_adhesion_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_LTi_agents<<<gridSize, blockSize>>>(d_xmachine_memory_LTi_values, d_LTis_adhesion, d_LTis_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_LTi_list* d_LTis_temp = d_LTis_adhesion;
	d_LTis_adhesion = d_LTis_swap;
	d_LTis_swap = d_LTis_temp;	
}

void sort_LTos_no_expression(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTo_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_LTo_no_expression_count); 
	gridSize = (h_xmachine_memory_LTo_no_expression_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_LTo_keys, d_xmachine_memory_LTo_values, d_LTos_no_expression);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_LTo_keys),  thrust::device_pointer_cast(d_xmachine_memory_LTo_keys) + h_xmachine_memory_LTo_no_expression_count,  thrust::device_pointer_cast(d_xmachine_memory_LTo_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_LTo_agents, no_sm, h_xmachine_memory_LTo_no_expression_count); 
	gridSize = (h_xmachine_memory_LTo_no_expression_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_LTo_agents<<<gridSize, blockSize>>>(d_xmachine_memory_LTo_values, d_LTos_no_expression, d_LTos_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_LTo_list* d_LTos_temp = d_LTos_no_expression;
	d_LTos_no_expression = d_LTos_swap;
	d_LTos_swap = d_LTos_temp;	
}

void sort_LTos_expression(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTo_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_LTo_expression_count); 
	gridSize = (h_xmachine_memory_LTo_expression_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_LTo_keys, d_xmachine_memory_LTo_values, d_LTos_expression);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_LTo_keys),  thrust::device_pointer_cast(d_xmachine_memory_LTo_keys) + h_xmachine_memory_LTo_expression_count,  thrust::device_pointer_cast(d_xmachine_memory_LTo_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_LTo_agents, no_sm, h_xmachine_memory_LTo_expression_count); 
	gridSize = (h_xmachine_memory_LTo_expression_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_LTo_agents<<<gridSize, blockSize>>>(d_xmachine_memory_LTo_values, d_LTos_expression, d_LTos_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_LTo_list* d_LTos_temp = d_LTos_expression;
	d_LTos_expression = d_LTos_swap;
	d_LTos_swap = d_LTos_temp;	
}

void sort_LTos_adhesion_upregulation(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTo_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_LTo_adhesion_upregulation_count); 
	gridSize = (h_xmachine_memory_LTo_adhesion_upregulation_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_LTo_keys, d_xmachine_memory_LTo_values, d_LTos_adhesion_upregulation);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_LTo_keys),  thrust::device_pointer_cast(d_xmachine_memory_LTo_keys) + h_xmachine_memory_LTo_adhesion_upregulation_count,  thrust::device_pointer_cast(d_xmachine_memory_LTo_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_LTo_agents, no_sm, h_xmachine_memory_LTo_adhesion_upregulation_count); 
	gridSize = (h_xmachine_memory_LTo_adhesion_upregulation_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_LTo_agents<<<gridSize, blockSize>>>(d_xmachine_memory_LTo_values, d_LTos_adhesion_upregulation, d_LTos_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_LTo_list* d_LTos_temp = d_LTos_adhesion_upregulation;
	d_LTos_adhesion_upregulation = d_LTos_swap;
	d_LTos_swap = d_LTos_temp;	
}

void sort_LTos_chemokine_upregulation(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTo_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_LTo_chemokine_upregulation_count); 
	gridSize = (h_xmachine_memory_LTo_chemokine_upregulation_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_LTo_keys, d_xmachine_memory_LTo_values, d_LTos_chemokine_upregulation);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_LTo_keys),  thrust::device_pointer_cast(d_xmachine_memory_LTo_keys) + h_xmachine_memory_LTo_chemokine_upregulation_count,  thrust::device_pointer_cast(d_xmachine_memory_LTo_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_LTo_agents, no_sm, h_xmachine_memory_LTo_chemokine_upregulation_count); 
	gridSize = (h_xmachine_memory_LTo_chemokine_upregulation_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_LTo_agents<<<gridSize, blockSize>>>(d_xmachine_memory_LTo_values, d_LTos_chemokine_upregulation, d_LTos_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_LTo_list* d_LTos_temp = d_LTos_chemokine_upregulation;
	d_LTos_chemokine_upregulation = d_LTos_swap;
	d_LTos_swap = d_LTos_temp;	
}

void sort_LTos_mature(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTo_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_LTo_mature_count); 
	gridSize = (h_xmachine_memory_LTo_mature_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_LTo_keys, d_xmachine_memory_LTo_values, d_LTos_mature);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_LTo_keys),  thrust::device_pointer_cast(d_xmachine_memory_LTo_keys) + h_xmachine_memory_LTo_mature_count,  thrust::device_pointer_cast(d_xmachine_memory_LTo_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_LTo_agents, no_sm, h_xmachine_memory_LTo_mature_count); 
	gridSize = (h_xmachine_memory_LTo_mature_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_LTo_agents<<<gridSize, blockSize>>>(d_xmachine_memory_LTo_values, d_LTos_mature, d_LTos_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_LTo_list* d_LTos_temp = d_LTos_mature;
	d_LTos_mature = d_LTos_swap;
	d_LTos_swap = d_LTos_temp;	
}

void sort_LTos_downregulated(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTo_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_LTo_downregulated_count); 
	gridSize = (h_xmachine_memory_LTo_downregulated_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_LTo_keys, d_xmachine_memory_LTo_values, d_LTos_downregulated);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_LTo_keys),  thrust::device_pointer_cast(d_xmachine_memory_LTo_keys) + h_xmachine_memory_LTo_downregulated_count,  thrust::device_pointer_cast(d_xmachine_memory_LTo_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_LTo_agents, no_sm, h_xmachine_memory_LTo_downregulated_count); 
	gridSize = (h_xmachine_memory_LTo_downregulated_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_LTo_agents<<<gridSize, blockSize>>>(d_xmachine_memory_LTo_values, d_LTos_downregulated, d_LTos_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_LTo_list* d_LTos_temp = d_LTos_downregulated;
	d_LTos_downregulated = d_LTos_swap;
	d_LTos_swap = d_LTos_temp;	
}


void cleanup(){

    /* Call all exit functions */
	

	/* Agent data free*/
	
	/* LTin Agent variables */
	gpuErrchk(cudaFree(d_LTins));
	gpuErrchk(cudaFree(d_LTins_swap));
	gpuErrchk(cudaFree(d_LTins_new));
	
	free( h_LTins_ltin_random_movement);
	gpuErrchk(cudaFree(d_LTins_ltin_random_movement));
	
	free( h_LTins_stable_contact);
	gpuErrchk(cudaFree(d_LTins_stable_contact));
	
	free( h_LTins_localised_movement);
	gpuErrchk(cudaFree(d_LTins_localised_movement));
	
	/* LTi Agent variables */
	gpuErrchk(cudaFree(d_LTis));
	gpuErrchk(cudaFree(d_LTis_swap));
	gpuErrchk(cudaFree(d_LTis_new));
	
	free( h_LTis_lti_random_movement);
	gpuErrchk(cudaFree(d_LTis_lti_random_movement));
	
	free( h_LTis_responding);
	gpuErrchk(cudaFree(d_LTis_responding));
	
	free( h_LTis_contact);
	gpuErrchk(cudaFree(d_LTis_contact));
	
	free( h_LTis_adhesion);
	gpuErrchk(cudaFree(d_LTis_adhesion));
	
	/* LTo Agent variables */
	gpuErrchk(cudaFree(d_LTos));
	gpuErrchk(cudaFree(d_LTos_swap));
	gpuErrchk(cudaFree(d_LTos_new));
	
	free( h_LTos_no_expression);
	gpuErrchk(cudaFree(d_LTos_no_expression));
	
	free( h_LTos_expression);
	gpuErrchk(cudaFree(d_LTos_expression));
	
	free( h_LTos_adhesion_upregulation);
	gpuErrchk(cudaFree(d_LTos_adhesion_upregulation));
	
	free( h_LTos_chemokine_upregulation);
	gpuErrchk(cudaFree(d_LTos_chemokine_upregulation));
	
	free( h_LTos_mature);
	gpuErrchk(cudaFree(d_LTos_mature));
	
	free( h_LTos_downregulated);
	gpuErrchk(cudaFree(d_LTos_downregulated));
	

	/* Message data free */
	
	/* location Message variables */
	free( h_locations);
	gpuErrchk(cudaFree(d_locations));
	gpuErrchk(cudaFree(d_locations_swap));
	
  
  /* CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamDestroy(stream1));
  gpuErrchk(cudaStreamDestroy(stream2));

  /* CUDA Event Timers for Instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventDestroy(instrument_iteration_start);
	cudaEventDestroy(instrument_iteration_stop);
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEventDestroy(instrument_start);
	cudaEventDestroy(instrument_stop);
#endif
}

void singleIteration(){

#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventRecord(instrument_iteration_start);
#endif

	/* set all non partitioned and spatial partitioned message counts to 0*/
	h_message_location_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_location_count, &h_message_location_count, sizeof(int)));
	

	/* Call agent functions in order iterating through the layer functions */
	
	/* Layer 1*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	LTi_lti_random_move(stream1);
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: LTi_lti_random_move = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	LTin_ltin_random_move(stream2);
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: LTin_ltin_random_move = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
    
    /* Call all step functions */
	

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("agent_LTin_ltin_random_movement_count: %u\n",get_agent_LTin_ltin_random_movement_count());
	
		printf("agent_LTin_stable_contact_count: %u\n",get_agent_LTin_stable_contact_count());
	
		printf("agent_LTin_localised_movement_count: %u\n",get_agent_LTin_localised_movement_count());
	
		printf("agent_LTi_lti_random_movement_count: %u\n",get_agent_LTi_lti_random_movement_count());
	
		printf("agent_LTi_responding_count: %u\n",get_agent_LTi_responding_count());
	
		printf("agent_LTi_contact_count: %u\n",get_agent_LTi_contact_count());
	
		printf("agent_LTi_adhesion_count: %u\n",get_agent_LTi_adhesion_count());
	
		printf("agent_LTo_no_expression_count: %u\n",get_agent_LTo_no_expression_count());
	
		printf("agent_LTo_expression_count: %u\n",get_agent_LTo_expression_count());
	
		printf("agent_LTo_adhesion_upregulation_count: %u\n",get_agent_LTo_adhesion_upregulation_count());
	
		printf("agent_LTo_chemokine_upregulation_count: %u\n",get_agent_LTo_chemokine_upregulation_count());
	
		printf("agent_LTo_mature_count: %u\n",get_agent_LTo_mature_count());
	
		printf("agent_LTo_downregulated_count: %u\n",get_agent_LTo_downregulated_count());
	
#endif

#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventRecord(instrument_iteration_stop);
	cudaEventSynchronize(instrument_iteration_stop);
	cudaEventElapsedTime(&instrument_iteration_milliseconds, instrument_iteration_start, instrument_iteration_stop);
	printf("Instrumentation: Iteration Time = %f (ms)\n", instrument_iteration_milliseconds);
#endif
}

/* Environment functions */

//host constant declaration
int h_env_LTI_CELL_SIZE;
int h_env_LTO_CELL_SIZE;
int h_env_adhesion_distance_threshold;
float h_env_CHEMO_THRESHOLD;
int h_env_CHEMO_CURVE_ADJUST;
float h_env_CHEMO_UPPER_ADJUST;
float h_env_CHEMO_LOWER_ADJUST;
float h_env_INCREASE_CHEMO_EXPRESSION;
float h_env_INITIAL_ADHESION;
float h_env_ADHESION_SLOPE;
float h_env_ADHESION_INCREMENT;
float h_env_MAX_ADHESION_PROBABILITY;
int h_env_INITIAL_CIRCUMFERENCE;
int h_env_MAXIMUM_CIRCUMFERENCE;
int h_env_INITIAL_LENGTH;
int h_env_MAXIMUM_LENGTH;
float h_env_STROMAL_CELL_DENSITY;
int h_env_GROWTH_TIME;
float h_env_PERCENT_LTIN_FROM_FACS;
float h_env_PERCENT_LTI_FROM_FACS;


//constant setter
void set_LTI_CELL_SIZE(int* h_LTI_CELL_SIZE){
    gpuErrchk(cudaMemcpyToSymbol(LTI_CELL_SIZE, h_LTI_CELL_SIZE, sizeof(int)));
    memcpy(&h_env_LTI_CELL_SIZE, h_LTI_CELL_SIZE,sizeof(int));
}


//constant getter
const int* get_LTI_CELL_SIZE(){
    return &h_env_LTI_CELL_SIZE;
}


//constant setter
void set_LTO_CELL_SIZE(int* h_LTO_CELL_SIZE){
    gpuErrchk(cudaMemcpyToSymbol(LTO_CELL_SIZE, h_LTO_CELL_SIZE, sizeof(int)));
    memcpy(&h_env_LTO_CELL_SIZE, h_LTO_CELL_SIZE,sizeof(int));
}


//constant getter
const int* get_LTO_CELL_SIZE(){
    return &h_env_LTO_CELL_SIZE;
}


//constant setter
void set_adhesion_distance_threshold(int* h_adhesion_distance_threshold){
    gpuErrchk(cudaMemcpyToSymbol(adhesion_distance_threshold, h_adhesion_distance_threshold, sizeof(int)));
    memcpy(&h_env_adhesion_distance_threshold, h_adhesion_distance_threshold,sizeof(int));
}


//constant getter
const int* get_adhesion_distance_threshold(){
    return &h_env_adhesion_distance_threshold;
}


//constant setter
void set_CHEMO_THRESHOLD(float* h_CHEMO_THRESHOLD){
    gpuErrchk(cudaMemcpyToSymbol(CHEMO_THRESHOLD, h_CHEMO_THRESHOLD, sizeof(float)));
    memcpy(&h_env_CHEMO_THRESHOLD, h_CHEMO_THRESHOLD,sizeof(float));
}


//constant getter
const float* get_CHEMO_THRESHOLD(){
    return &h_env_CHEMO_THRESHOLD;
}


//constant setter
void set_CHEMO_CURVE_ADJUST(int* h_CHEMO_CURVE_ADJUST){
    gpuErrchk(cudaMemcpyToSymbol(CHEMO_CURVE_ADJUST, h_CHEMO_CURVE_ADJUST, sizeof(int)));
    memcpy(&h_env_CHEMO_CURVE_ADJUST, h_CHEMO_CURVE_ADJUST,sizeof(int));
}


//constant getter
const int* get_CHEMO_CURVE_ADJUST(){
    return &h_env_CHEMO_CURVE_ADJUST;
}


//constant setter
void set_CHEMO_UPPER_ADJUST(float* h_CHEMO_UPPER_ADJUST){
    gpuErrchk(cudaMemcpyToSymbol(CHEMO_UPPER_ADJUST, h_CHEMO_UPPER_ADJUST, sizeof(float)));
    memcpy(&h_env_CHEMO_UPPER_ADJUST, h_CHEMO_UPPER_ADJUST,sizeof(float));
}


//constant getter
const float* get_CHEMO_UPPER_ADJUST(){
    return &h_env_CHEMO_UPPER_ADJUST;
}


//constant setter
void set_CHEMO_LOWER_ADJUST(float* h_CHEMO_LOWER_ADJUST){
    gpuErrchk(cudaMemcpyToSymbol(CHEMO_LOWER_ADJUST, h_CHEMO_LOWER_ADJUST, sizeof(float)));
    memcpy(&h_env_CHEMO_LOWER_ADJUST, h_CHEMO_LOWER_ADJUST,sizeof(float));
}


//constant getter
const float* get_CHEMO_LOWER_ADJUST(){
    return &h_env_CHEMO_LOWER_ADJUST;
}


//constant setter
void set_INCREASE_CHEMO_EXPRESSION(float* h_INCREASE_CHEMO_EXPRESSION){
    gpuErrchk(cudaMemcpyToSymbol(INCREASE_CHEMO_EXPRESSION, h_INCREASE_CHEMO_EXPRESSION, sizeof(float)));
    memcpy(&h_env_INCREASE_CHEMO_EXPRESSION, h_INCREASE_CHEMO_EXPRESSION,sizeof(float));
}


//constant getter
const float* get_INCREASE_CHEMO_EXPRESSION(){
    return &h_env_INCREASE_CHEMO_EXPRESSION;
}


//constant setter
void set_INITIAL_ADHESION(float* h_INITIAL_ADHESION){
    gpuErrchk(cudaMemcpyToSymbol(INITIAL_ADHESION, h_INITIAL_ADHESION, sizeof(float)));
    memcpy(&h_env_INITIAL_ADHESION, h_INITIAL_ADHESION,sizeof(float));
}


//constant getter
const float* get_INITIAL_ADHESION(){
    return &h_env_INITIAL_ADHESION;
}


//constant setter
void set_ADHESION_SLOPE(float* h_ADHESION_SLOPE){
    gpuErrchk(cudaMemcpyToSymbol(ADHESION_SLOPE, h_ADHESION_SLOPE, sizeof(float)));
    memcpy(&h_env_ADHESION_SLOPE, h_ADHESION_SLOPE,sizeof(float));
}


//constant getter
const float* get_ADHESION_SLOPE(){
    return &h_env_ADHESION_SLOPE;
}


//constant setter
void set_ADHESION_INCREMENT(float* h_ADHESION_INCREMENT){
    gpuErrchk(cudaMemcpyToSymbol(ADHESION_INCREMENT, h_ADHESION_INCREMENT, sizeof(float)));
    memcpy(&h_env_ADHESION_INCREMENT, h_ADHESION_INCREMENT,sizeof(float));
}


//constant getter
const float* get_ADHESION_INCREMENT(){
    return &h_env_ADHESION_INCREMENT;
}


//constant setter
void set_MAX_ADHESION_PROBABILITY(float* h_MAX_ADHESION_PROBABILITY){
    gpuErrchk(cudaMemcpyToSymbol(MAX_ADHESION_PROBABILITY, h_MAX_ADHESION_PROBABILITY, sizeof(float)));
    memcpy(&h_env_MAX_ADHESION_PROBABILITY, h_MAX_ADHESION_PROBABILITY,sizeof(float));
}


//constant getter
const float* get_MAX_ADHESION_PROBABILITY(){
    return &h_env_MAX_ADHESION_PROBABILITY;
}


//constant setter
void set_INITIAL_CIRCUMFERENCE(int* h_INITIAL_CIRCUMFERENCE){
    gpuErrchk(cudaMemcpyToSymbol(INITIAL_CIRCUMFERENCE, h_INITIAL_CIRCUMFERENCE, sizeof(int)));
    memcpy(&h_env_INITIAL_CIRCUMFERENCE, h_INITIAL_CIRCUMFERENCE,sizeof(int));
}


//constant getter
const int* get_INITIAL_CIRCUMFERENCE(){
    return &h_env_INITIAL_CIRCUMFERENCE;
}


//constant setter
void set_MAXIMUM_CIRCUMFERENCE(int* h_MAXIMUM_CIRCUMFERENCE){
    gpuErrchk(cudaMemcpyToSymbol(MAXIMUM_CIRCUMFERENCE, h_MAXIMUM_CIRCUMFERENCE, sizeof(int)));
    memcpy(&h_env_MAXIMUM_CIRCUMFERENCE, h_MAXIMUM_CIRCUMFERENCE,sizeof(int));
}


//constant getter
const int* get_MAXIMUM_CIRCUMFERENCE(){
    return &h_env_MAXIMUM_CIRCUMFERENCE;
}


//constant setter
void set_INITIAL_LENGTH(int* h_INITIAL_LENGTH){
    gpuErrchk(cudaMemcpyToSymbol(INITIAL_LENGTH, h_INITIAL_LENGTH, sizeof(int)));
    memcpy(&h_env_INITIAL_LENGTH, h_INITIAL_LENGTH,sizeof(int));
}


//constant getter
const int* get_INITIAL_LENGTH(){
    return &h_env_INITIAL_LENGTH;
}


//constant setter
void set_MAXIMUM_LENGTH(int* h_MAXIMUM_LENGTH){
    gpuErrchk(cudaMemcpyToSymbol(MAXIMUM_LENGTH, h_MAXIMUM_LENGTH, sizeof(int)));
    memcpy(&h_env_MAXIMUM_LENGTH, h_MAXIMUM_LENGTH,sizeof(int));
}


//constant getter
const int* get_MAXIMUM_LENGTH(){
    return &h_env_MAXIMUM_LENGTH;
}


//constant setter
void set_STROMAL_CELL_DENSITY(float* h_STROMAL_CELL_DENSITY){
    gpuErrchk(cudaMemcpyToSymbol(STROMAL_CELL_DENSITY, h_STROMAL_CELL_DENSITY, sizeof(float)));
    memcpy(&h_env_STROMAL_CELL_DENSITY, h_STROMAL_CELL_DENSITY,sizeof(float));
}


//constant getter
const float* get_STROMAL_CELL_DENSITY(){
    return &h_env_STROMAL_CELL_DENSITY;
}


//constant setter
void set_GROWTH_TIME(int* h_GROWTH_TIME){
    gpuErrchk(cudaMemcpyToSymbol(GROWTH_TIME, h_GROWTH_TIME, sizeof(int)));
    memcpy(&h_env_GROWTH_TIME, h_GROWTH_TIME,sizeof(int));
}


//constant getter
const int* get_GROWTH_TIME(){
    return &h_env_GROWTH_TIME;
}


//constant setter
void set_PERCENT_LTIN_FROM_FACS(float* h_PERCENT_LTIN_FROM_FACS){
    gpuErrchk(cudaMemcpyToSymbol(PERCENT_LTIN_FROM_FACS, h_PERCENT_LTIN_FROM_FACS, sizeof(float)));
    memcpy(&h_env_PERCENT_LTIN_FROM_FACS, h_PERCENT_LTIN_FROM_FACS,sizeof(float));
}


//constant getter
const float* get_PERCENT_LTIN_FROM_FACS(){
    return &h_env_PERCENT_LTIN_FROM_FACS;
}


//constant setter
void set_PERCENT_LTI_FROM_FACS(float* h_PERCENT_LTI_FROM_FACS){
    gpuErrchk(cudaMemcpyToSymbol(PERCENT_LTI_FROM_FACS, h_PERCENT_LTI_FROM_FACS, sizeof(float)));
    memcpy(&h_env_PERCENT_LTI_FROM_FACS, h_PERCENT_LTI_FROM_FACS,sizeof(float));
}


//constant getter
const float* get_PERCENT_LTI_FROM_FACS(){
    return &h_env_PERCENT_LTI_FROM_FACS;
}



/* Agent data access functions*/

    
int get_agent_LTin_MAX_count(){
    return xmachine_memory_LTin_MAX;
}


int get_agent_LTin_ltin_random_movement_count(){
	//continuous agent
	return h_xmachine_memory_LTin_ltin_random_movement_count;
	
}

xmachine_memory_LTin_list* get_device_LTin_ltin_random_movement_agents(){
	return d_LTins_ltin_random_movement;
}

xmachine_memory_LTin_list* get_host_LTin_ltin_random_movement_agents(){
	return h_LTins_ltin_random_movement;
}

int get_agent_LTin_stable_contact_count(){
	//continuous agent
	return h_xmachine_memory_LTin_stable_contact_count;
	
}

xmachine_memory_LTin_list* get_device_LTin_stable_contact_agents(){
	return d_LTins_stable_contact;
}

xmachine_memory_LTin_list* get_host_LTin_stable_contact_agents(){
	return h_LTins_stable_contact;
}

int get_agent_LTin_localised_movement_count(){
	//continuous agent
	return h_xmachine_memory_LTin_localised_movement_count;
	
}

xmachine_memory_LTin_list* get_device_LTin_localised_movement_agents(){
	return d_LTins_localised_movement;
}

xmachine_memory_LTin_list* get_host_LTin_localised_movement_agents(){
	return h_LTins_localised_movement;
}

    
int get_agent_LTi_MAX_count(){
    return xmachine_memory_LTi_MAX;
}


int get_agent_LTi_lti_random_movement_count(){
	//continuous agent
	return h_xmachine_memory_LTi_lti_random_movement_count;
	
}

xmachine_memory_LTi_list* get_device_LTi_lti_random_movement_agents(){
	return d_LTis_lti_random_movement;
}

xmachine_memory_LTi_list* get_host_LTi_lti_random_movement_agents(){
	return h_LTis_lti_random_movement;
}

int get_agent_LTi_responding_count(){
	//continuous agent
	return h_xmachine_memory_LTi_responding_count;
	
}

xmachine_memory_LTi_list* get_device_LTi_responding_agents(){
	return d_LTis_responding;
}

xmachine_memory_LTi_list* get_host_LTi_responding_agents(){
	return h_LTis_responding;
}

int get_agent_LTi_contact_count(){
	//continuous agent
	return h_xmachine_memory_LTi_contact_count;
	
}

xmachine_memory_LTi_list* get_device_LTi_contact_agents(){
	return d_LTis_contact;
}

xmachine_memory_LTi_list* get_host_LTi_contact_agents(){
	return h_LTis_contact;
}

int get_agent_LTi_adhesion_count(){
	//continuous agent
	return h_xmachine_memory_LTi_adhesion_count;
	
}

xmachine_memory_LTi_list* get_device_LTi_adhesion_agents(){
	return d_LTis_adhesion;
}

xmachine_memory_LTi_list* get_host_LTi_adhesion_agents(){
	return h_LTis_adhesion;
}

    
int get_agent_LTo_MAX_count(){
    return xmachine_memory_LTo_MAX;
}


int get_agent_LTo_no_expression_count(){
	//continuous agent
	return h_xmachine_memory_LTo_no_expression_count;
	
}

xmachine_memory_LTo_list* get_device_LTo_no_expression_agents(){
	return d_LTos_no_expression;
}

xmachine_memory_LTo_list* get_host_LTo_no_expression_agents(){
	return h_LTos_no_expression;
}

int get_agent_LTo_expression_count(){
	//continuous agent
	return h_xmachine_memory_LTo_expression_count;
	
}

xmachine_memory_LTo_list* get_device_LTo_expression_agents(){
	return d_LTos_expression;
}

xmachine_memory_LTo_list* get_host_LTo_expression_agents(){
	return h_LTos_expression;
}

int get_agent_LTo_adhesion_upregulation_count(){
	//continuous agent
	return h_xmachine_memory_LTo_adhesion_upregulation_count;
	
}

xmachine_memory_LTo_list* get_device_LTo_adhesion_upregulation_agents(){
	return d_LTos_adhesion_upregulation;
}

xmachine_memory_LTo_list* get_host_LTo_adhesion_upregulation_agents(){
	return h_LTos_adhesion_upregulation;
}

int get_agent_LTo_chemokine_upregulation_count(){
	//continuous agent
	return h_xmachine_memory_LTo_chemokine_upregulation_count;
	
}

xmachine_memory_LTo_list* get_device_LTo_chemokine_upregulation_agents(){
	return d_LTos_chemokine_upregulation;
}

xmachine_memory_LTo_list* get_host_LTo_chemokine_upregulation_agents(){
	return h_LTos_chemokine_upregulation;
}

int get_agent_LTo_mature_count(){
	//continuous agent
	return h_xmachine_memory_LTo_mature_count;
	
}

xmachine_memory_LTo_list* get_device_LTo_mature_agents(){
	return d_LTos_mature;
}

xmachine_memory_LTo_list* get_host_LTo_mature_agents(){
	return h_LTos_mature;
}

int get_agent_LTo_downregulated_count(){
	//continuous agent
	return h_xmachine_memory_LTo_downregulated_count;
	
}

xmachine_memory_LTo_list* get_device_LTo_downregulated_agents(){
	return d_LTos_downregulated;
}

xmachine_memory_LTo_list* get_host_LTo_downregulated_agents(){
	return h_LTos_downregulated;
}



/*  Analytics Functions */

float reduce_LTin_ltin_random_movement_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTins_ltin_random_movement->x),  thrust::device_pointer_cast(d_LTins_ltin_random_movement->x) + h_xmachine_memory_LTin_ltin_random_movement_count);
}

float reduce_LTin_ltin_random_movement_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTins_ltin_random_movement->y),  thrust::device_pointer_cast(d_LTins_ltin_random_movement->y) + h_xmachine_memory_LTin_ltin_random_movement_count);
}

float reduce_LTin_stable_contact_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTins_stable_contact->x),  thrust::device_pointer_cast(d_LTins_stable_contact->x) + h_xmachine_memory_LTin_stable_contact_count);
}

float reduce_LTin_stable_contact_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTins_stable_contact->y),  thrust::device_pointer_cast(d_LTins_stable_contact->y) + h_xmachine_memory_LTin_stable_contact_count);
}

float reduce_LTin_localised_movement_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTins_localised_movement->x),  thrust::device_pointer_cast(d_LTins_localised_movement->x) + h_xmachine_memory_LTin_localised_movement_count);
}

float reduce_LTin_localised_movement_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTins_localised_movement->y),  thrust::device_pointer_cast(d_LTins_localised_movement->y) + h_xmachine_memory_LTin_localised_movement_count);
}

float reduce_LTi_lti_random_movement_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTis_lti_random_movement->x),  thrust::device_pointer_cast(d_LTis_lti_random_movement->x) + h_xmachine_memory_LTi_lti_random_movement_count);
}

float reduce_LTi_lti_random_movement_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTis_lti_random_movement->y),  thrust::device_pointer_cast(d_LTis_lti_random_movement->y) + h_xmachine_memory_LTi_lti_random_movement_count);
}

float reduce_LTi_responding_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTis_responding->x),  thrust::device_pointer_cast(d_LTis_responding->x) + h_xmachine_memory_LTi_responding_count);
}

float reduce_LTi_responding_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTis_responding->y),  thrust::device_pointer_cast(d_LTis_responding->y) + h_xmachine_memory_LTi_responding_count);
}

float reduce_LTi_contact_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTis_contact->x),  thrust::device_pointer_cast(d_LTis_contact->x) + h_xmachine_memory_LTi_contact_count);
}

float reduce_LTi_contact_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTis_contact->y),  thrust::device_pointer_cast(d_LTis_contact->y) + h_xmachine_memory_LTi_contact_count);
}

float reduce_LTi_adhesion_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTis_adhesion->x),  thrust::device_pointer_cast(d_LTis_adhesion->x) + h_xmachine_memory_LTi_adhesion_count);
}

float reduce_LTi_adhesion_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTis_adhesion->y),  thrust::device_pointer_cast(d_LTis_adhesion->y) + h_xmachine_memory_LTi_adhesion_count);
}

float reduce_LTo_no_expression_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTos_no_expression->x),  thrust::device_pointer_cast(d_LTos_no_expression->x) + h_xmachine_memory_LTo_no_expression_count);
}

float reduce_LTo_no_expression_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTos_no_expression->y),  thrust::device_pointer_cast(d_LTos_no_expression->y) + h_xmachine_memory_LTo_no_expression_count);
}

float reduce_LTo_expression_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTos_expression->x),  thrust::device_pointer_cast(d_LTos_expression->x) + h_xmachine_memory_LTo_expression_count);
}

float reduce_LTo_expression_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTos_expression->y),  thrust::device_pointer_cast(d_LTos_expression->y) + h_xmachine_memory_LTo_expression_count);
}

float reduce_LTo_adhesion_upregulation_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTos_adhesion_upregulation->x),  thrust::device_pointer_cast(d_LTos_adhesion_upregulation->x) + h_xmachine_memory_LTo_adhesion_upregulation_count);
}

float reduce_LTo_adhesion_upregulation_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTos_adhesion_upregulation->y),  thrust::device_pointer_cast(d_LTos_adhesion_upregulation->y) + h_xmachine_memory_LTo_adhesion_upregulation_count);
}

float reduce_LTo_chemokine_upregulation_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTos_chemokine_upregulation->x),  thrust::device_pointer_cast(d_LTos_chemokine_upregulation->x) + h_xmachine_memory_LTo_chemokine_upregulation_count);
}

float reduce_LTo_chemokine_upregulation_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTos_chemokine_upregulation->y),  thrust::device_pointer_cast(d_LTos_chemokine_upregulation->y) + h_xmachine_memory_LTo_chemokine_upregulation_count);
}

float reduce_LTo_mature_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTos_mature->x),  thrust::device_pointer_cast(d_LTos_mature->x) + h_xmachine_memory_LTo_mature_count);
}

float reduce_LTo_mature_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTos_mature->y),  thrust::device_pointer_cast(d_LTos_mature->y) + h_xmachine_memory_LTo_mature_count);
}

float reduce_LTo_downregulated_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTos_downregulated->x),  thrust::device_pointer_cast(d_LTos_downregulated->x) + h_xmachine_memory_LTo_downregulated_count);
}

float reduce_LTo_downregulated_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_LTos_downregulated->y),  thrust::device_pointer_cast(d_LTos_downregulated->y) + h_xmachine_memory_LTo_downregulated_count);
}




/* Agent functions */


	
/* Shared memory size calculator for agent function */
int LTin_ltin_random_move_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** LTin_ltin_random_move
 * Agent function prototype for ltin_random_move function of LTin agent
 */
void LTin_ltin_random_move(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_LTin_ltin_random_movement_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_LTin_ltin_random_movement_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_LTin_list* LTins_ltin_random_movement_temp = d_LTins;
	d_LTins = d_LTins_ltin_random_movement;
	d_LTins_ltin_random_movement = LTins_ltin_random_movement_temp;
	//set working count to current state count
	h_xmachine_memory_LTin_count = h_xmachine_memory_LTin_ltin_random_movement_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_LTin_count, &h_xmachine_memory_LTin_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_LTin_ltin_random_movement_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_LTin_ltin_random_movement_count, &h_xmachine_memory_LTin_ltin_random_movement_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_location_count + h_xmachine_memory_LTin_count > xmachine_message_location_MAX){
		printf("Error: Buffer size of location message will be exceeded in function ltin_random_move\n");
		exit(0);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_ltin_random_move, LTin_ltin_random_move_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = LTin_ltin_random_move_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_location_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_location_output_type, &h_message_location_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (ltin_random_move)
	//Reallocate   : false
	//Input        : 
	//Output       : location
	//Agent Output : 
	GPUFLAME_ltin_random_move<<<g, b, sm_size, stream>>>(d_LTins, d_locations);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_location_count += h_xmachine_memory_LTin_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_location_count, &h_message_location_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_LTin_ltin_random_movement_count+h_xmachine_memory_LTin_count > xmachine_memory_LTin_MAX){
		printf("Error: Buffer size of ltin_random_move agents in state ltin_random_movement will be exceeded moving working agents to next state in function ltin_random_move\n");
		exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_LTin_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_LTin_Agents<<<gridSize, blockSize, 0, stream>>>(d_LTins_ltin_random_movement, d_LTins, h_xmachine_memory_LTin_ltin_random_movement_count, h_xmachine_memory_LTin_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_LTin_ltin_random_movement_count += h_xmachine_memory_LTin_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_LTin_ltin_random_movement_count, &h_xmachine_memory_LTin_ltin_random_movement_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int LTi_lti_random_move_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** LTi_lti_random_move
 * Agent function prototype for lti_random_move function of LTi agent
 */
void LTi_lti_random_move(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_LTi_lti_random_movement_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_LTi_lti_random_movement_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_LTi_list* LTis_lti_random_movement_temp = d_LTis;
	d_LTis = d_LTis_lti_random_movement;
	d_LTis_lti_random_movement = LTis_lti_random_movement_temp;
	//set working count to current state count
	h_xmachine_memory_LTi_count = h_xmachine_memory_LTi_lti_random_movement_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_LTi_count, &h_xmachine_memory_LTi_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_LTi_lti_random_movement_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_LTi_lti_random_movement_count, &h_xmachine_memory_LTi_lti_random_movement_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_location_count + h_xmachine_memory_LTi_count > xmachine_message_location_MAX){
		printf("Error: Buffer size of location message will be exceeded in function lti_random_move\n");
		exit(0);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_lti_random_move, LTi_lti_random_move_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = LTi_lti_random_move_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_location_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_location_output_type, &h_message_location_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (lti_random_move)
	//Reallocate   : false
	//Input        : 
	//Output       : location
	//Agent Output : 
	GPUFLAME_lti_random_move<<<g, b, sm_size, stream>>>(d_LTis, d_locations);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_location_count += h_xmachine_memory_LTi_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_location_count, &h_message_location_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_LTi_lti_random_movement_count+h_xmachine_memory_LTi_count > xmachine_memory_LTi_MAX){
		printf("Error: Buffer size of lti_random_move agents in state lti_random_movement will be exceeded moving working agents to next state in function lti_random_move\n");
		exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_LTi_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_LTi_Agents<<<gridSize, blockSize, 0, stream>>>(d_LTis_lti_random_movement, d_LTis, h_xmachine_memory_LTi_lti_random_movement_count, h_xmachine_memory_LTi_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_LTi_lti_random_movement_count += h_xmachine_memory_LTi_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_LTi_lti_random_movement_count, &h_xmachine_memory_LTi_lti_random_movement_count, sizeof(int)));	
	
	
}


 
extern void reset_LTin_ltin_random_movement_count()
{
    h_xmachine_memory_LTin_ltin_random_movement_count = 0;
}
 
extern void reset_LTin_stable_contact_count()
{
    h_xmachine_memory_LTin_stable_contact_count = 0;
}
 
extern void reset_LTin_localised_movement_count()
{
    h_xmachine_memory_LTin_localised_movement_count = 0;
}
 
extern void reset_LTi_lti_random_movement_count()
{
    h_xmachine_memory_LTi_lti_random_movement_count = 0;
}
 
extern void reset_LTi_responding_count()
{
    h_xmachine_memory_LTi_responding_count = 0;
}
 
extern void reset_LTi_contact_count()
{
    h_xmachine_memory_LTi_contact_count = 0;
}
 
extern void reset_LTi_adhesion_count()
{
    h_xmachine_memory_LTi_adhesion_count = 0;
}
 
extern void reset_LTo_no_expression_count()
{
    h_xmachine_memory_LTo_no_expression_count = 0;
}
 
extern void reset_LTo_expression_count()
{
    h_xmachine_memory_LTo_expression_count = 0;
}
 
extern void reset_LTo_adhesion_upregulation_count()
{
    h_xmachine_memory_LTo_adhesion_upregulation_count = 0;
}
 
extern void reset_LTo_chemokine_upregulation_count()
{
    h_xmachine_memory_LTo_chemokine_upregulation_count = 0;
}
 
extern void reset_LTo_mature_count()
{
    h_xmachine_memory_LTo_mature_count = 0;
}
 
extern void reset_LTo_downregulated_count()
{
    h_xmachine_memory_LTo_downregulated_count = 0;
}
