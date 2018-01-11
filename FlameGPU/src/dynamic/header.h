
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

#ifndef __HEADER
#define __HEADER
#define GLM_FORCE_NO_CTOR_INIT
#include <glm/glm.hpp>

/* General standard definitions */
//Threads per block (agents per block)
#define THREADS_PER_TILE 64
//Definition for any agent function or helper function
#define __FLAME_GPU_FUNC__ __device__
//Definition for a function used to initialise environment variables
#define __FLAME_GPU_INIT_FUNC__
#define __FLAME_GPU_STEP_FUNC__
#define __FLAME_GPU_EXIT_FUNC__

#define USE_CUDA_STREAMS
#define FAST_ATOMIC_SORTING

typedef unsigned int uint;


	

/* Agent population size definitions must be a multiple of THREADS_PER_TILE (default 64) */
//Maximum buffer size (largest agent buffer size)
#define buffer_size_MAX 1024

//Maximum population size of xmachine_memory_LTin
#define xmachine_memory_LTin_MAX 1024

//Maximum population size of xmachine_memory_LTi
#define xmachine_memory_LTi_MAX 1024

//Maximum population size of xmachine_memory_LTo
#define xmachine_memory_LTo_MAX 1024
  
  
/* Message population size definitions */
//Maximum population size of xmachine_mmessage_location
#define xmachine_message_location_MAX 1024



/* Spatial partitioning grid size definitions */
  
  
/* enum types */

/**
 * MESSAGE_OUTPUT used for all continuous messaging
 */
enum MESSAGE_OUTPUT{
	single_message,
	optional_message,
};

/**
 * AGENT_TYPE used for templates device message functions
 */
enum AGENT_TYPE{
	CONTINUOUS,
	DISCRETE_2D
};


/* Agent structures */

/** struct xmachine_memory_LTin
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_LTin
{
    float x;    /**< X-machine memory variable x of type float.*/
    float y;    /**< X-machine memory variable y of type float.*/
};

/** struct xmachine_memory_LTi
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_LTi
{
    float x;    /**< X-machine memory variable x of type float.*/
    float y;    /**< X-machine memory variable y of type float.*/
};

/** struct xmachine_memory_LTo
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_LTo
{
    float x;    /**< X-machine memory variable x of type float.*/
    float y;    /**< X-machine memory variable y of type float.*/
};



/* Message structures */

/** struct xmachine_message_location
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_location
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    float x;        /**< Message variable x of type float.*/  
    float y;        /**< Message variable y of type float.*/
};



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_LTin_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_LTin_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_LTin_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_LTin_MAX];  /**< Used during parallel prefix sum */
    
    float x [xmachine_memory_LTin_MAX];    /**< X-machine memory variable list x of type float.*/
    float y [xmachine_memory_LTin_MAX];    /**< X-machine memory variable list y of type float.*/
};

/** struct xmachine_memory_LTi_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_LTi_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_LTi_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_LTi_MAX];  /**< Used during parallel prefix sum */
    
    float x [xmachine_memory_LTi_MAX];    /**< X-machine memory variable list x of type float.*/
    float y [xmachine_memory_LTi_MAX];    /**< X-machine memory variable list y of type float.*/
};

/** struct xmachine_memory_LTo_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_LTo_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_LTo_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_LTo_MAX];  /**< Used during parallel prefix sum */
    
    float x [xmachine_memory_LTo_MAX];    /**< X-machine memory variable list x of type float.*/
    float y [xmachine_memory_LTo_MAX];    /**< X-machine memory variable list y of type float.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_location_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_location_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_location_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_location_MAX];  /**< Used during parallel prefix sum */
    
    float x [xmachine_message_location_MAX];    /**< Message memory variable list x of type float.*/
    float y [xmachine_message_location_MAX];    /**< Message memory variable list y of type float.*/
    
};



/* Spatially Partitioned Message boundary Matrices */



  /* Random */
  /** struct RNG_rand48
  *	structure used to hold list seeds
  */
  struct RNG_rand48
  {
  glm::uvec2 A, C;
  glm::uvec2 seeds[buffer_size_MAX];
  };


/** getOutputDir
* Gets the output directory of the simulation. This is the same as the 0.xml input directory.
* @return a const char pointer to string denoting the output directory
*/
const char* getOutputDir();

  /* Random Functions (usable in agent functions) implemented in FLAMEGPU_Kernels */

  /**
  * Templated random function using a DISCRETE_2D template calculates the agent index using a 2D block
  * which requires extra processing but will work for CONTINUOUS agents. Using a CONTINUOUS template will
  * not work for DISCRETE_2D agent.
  * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
  * @return			returns a random float value
  */
  template <int AGENT_TYPE> __FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);
/**
 * Non templated random function calls the templated version with DISCRETE_2D which will work in either case
 * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
 * @return			returns a random float value
 */
__FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);

/* Agent function prototypes */

/**
 * ltin_random_move FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_LTin. This represents a single agent instance and can be modified directly.
 * @param location_messages Pointer to output message list of type xmachine_message_location_list. Must be passed as an argument to the add_location_message function ??.
 */
__FLAME_GPU_FUNC__ int ltin_random_move(xmachine_memory_LTin* agent, xmachine_message_location_list* location_messages);

/**
 * lti_random_move FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_LTi. This represents a single agent instance and can be modified directly.
 * @param location_messages Pointer to output message list of type xmachine_message_location_list. Must be passed as an argument to the add_location_message function ??.
 */
__FLAME_GPU_FUNC__ int lti_random_move(xmachine_memory_LTi* agent, xmachine_message_location_list* location_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) location message implemented in FLAMEGPU_Kernels */

/** add_location_message
 * Function for all types of message partitioning
 * Adds a new location agent to the xmachine_memory_location_list list using a linear mapping
 * @param agents	xmachine_memory_location_list agent list
 * @param x	message variable of type float
 * @param y	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_location_message(xmachine_message_location_list* location_messages, float x, float y);
 
/** get_first_location_message
 * Get first message function for non partitioned (brute force) messages
 * @param location_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_location * get_first_location_message(xmachine_message_location_list* location_messages);

/** get_next_location_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param location_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_location * get_next_location_message(xmachine_message_location* current, xmachine_message_location_list* location_messages);
  
  
  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */

/** add_LTin_agent
 * Adds a new continuous valued LTin agent to the xmachine_memory_LTin_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_LTin_list agent list
 * @param x	agent agent variable of type float
 * @param y	agent agent variable of type float
 */
__FLAME_GPU_FUNC__ void add_LTin_agent(xmachine_memory_LTin_list* agents, float x, float y);

/** add_LTi_agent
 * Adds a new continuous valued LTi agent to the xmachine_memory_LTi_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_LTi_list agent list
 * @param x	agent agent variable of type float
 * @param y	agent agent variable of type float
 */
__FLAME_GPU_FUNC__ void add_LTi_agent(xmachine_memory_LTi_list* agents, float x, float y);

/** add_LTo_agent
 * Adds a new continuous valued LTo agent to the xmachine_memory_LTo_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_LTo_list agent list
 * @param x	agent agent variable of type float
 * @param y	agent agent variable of type float
 */
__FLAME_GPU_FUNC__ void add_LTo_agent(xmachine_memory_LTo_list* agents, float x, float y);


  
/* Simulation function prototypes implemented in simulation.cu */

/** initialise
 * Initialise the simulation. Allocated host and device memory. Reads the initial agent configuration from XML.
 * @param input	XML file path for agent initial configuration
 */
extern void initialise(char * input);

/** cleanup
 * Function cleans up any memory allocations on the host and device
 */
extern void cleanup();

/** singleIteration
 *	Performs a single iteration of the simulation. I.e. performs each agent function on each function layer in the correct order.
 */
extern void singleIteration();

/** saveIterationData
 * Reads the current agent data fromt he device and saves it to XML
 * @param	outputpath	file path to XML file used for output of agent data
 * @param	iteration_number
 * @param h_LTins Pointer to agent list on the host
 * @param d_LTins Pointer to agent list on the GPU device
 * @param h_xmachine_memory_LTin_count Pointer to agent counter
 * @param h_LTis Pointer to agent list on the host
 * @param d_LTis Pointer to agent list on the GPU device
 * @param h_xmachine_memory_LTi_count Pointer to agent counter
 * @param h_LTos Pointer to agent list on the host
 * @param d_LTos Pointer to agent list on the GPU device
 * @param h_xmachine_memory_LTo_count Pointer to agent counter
 */
extern void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_LTin_list* h_LTins_ltin_random_movement, xmachine_memory_LTin_list* d_LTins_ltin_random_movement, int h_xmachine_memory_LTin_ltin_random_movement_count,xmachine_memory_LTin_list* h_LTins_stable_contact, xmachine_memory_LTin_list* d_LTins_stable_contact, int h_xmachine_memory_LTin_stable_contact_count,xmachine_memory_LTin_list* h_LTins_localised_movement, xmachine_memory_LTin_list* d_LTins_localised_movement, int h_xmachine_memory_LTin_localised_movement_count,xmachine_memory_LTi_list* h_LTis_lti_random_movement, xmachine_memory_LTi_list* d_LTis_lti_random_movement, int h_xmachine_memory_LTi_lti_random_movement_count,xmachine_memory_LTi_list* h_LTis_responding, xmachine_memory_LTi_list* d_LTis_responding, int h_xmachine_memory_LTi_responding_count,xmachine_memory_LTi_list* h_LTis_contact, xmachine_memory_LTi_list* d_LTis_contact, int h_xmachine_memory_LTi_contact_count,xmachine_memory_LTi_list* h_LTis_adhesion, xmachine_memory_LTi_list* d_LTis_adhesion, int h_xmachine_memory_LTi_adhesion_count,xmachine_memory_LTo_list* h_LTos_no_expression, xmachine_memory_LTo_list* d_LTos_no_expression, int h_xmachine_memory_LTo_no_expression_count,xmachine_memory_LTo_list* h_LTos_expression, xmachine_memory_LTo_list* d_LTos_expression, int h_xmachine_memory_LTo_expression_count,xmachine_memory_LTo_list* h_LTos_adhesion_upregulation, xmachine_memory_LTo_list* d_LTos_adhesion_upregulation, int h_xmachine_memory_LTo_adhesion_upregulation_count,xmachine_memory_LTo_list* h_LTos_chemokine_upregulation, xmachine_memory_LTo_list* d_LTos_chemokine_upregulation, int h_xmachine_memory_LTo_chemokine_upregulation_count,xmachine_memory_LTo_list* h_LTos_mature, xmachine_memory_LTo_list* d_LTos_mature, int h_xmachine_memory_LTo_mature_count,xmachine_memory_LTo_list* h_LTos_downregulated, xmachine_memory_LTo_list* d_LTos_downregulated, int h_xmachine_memory_LTo_downregulated_count);


/** readInitialStates
 * Reads the current agent data from the device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_LTins Pointer to agent list on the host
 * @param h_xmachine_memory_LTin_count Pointer to agent counter
 * @param h_LTis Pointer to agent list on the host
 * @param h_xmachine_memory_LTi_count Pointer to agent counter
 * @param h_LTos Pointer to agent list on the host
 * @param h_xmachine_memory_LTo_count Pointer to agent counter
 */
extern void readInitialStates(char* inputpath, xmachine_memory_LTin_list* h_LTins, int* h_xmachine_memory_LTin_count,xmachine_memory_LTi_list* h_LTis, int* h_xmachine_memory_LTi_count,xmachine_memory_LTo_list* h_LTos, int* h_xmachine_memory_LTo_count);


/* Return functions used by external code to get agent data from device */

    
/** get_agent_LTin_MAX_count
 * Gets the max agent count for the LTin agent type 
 * @return		the maximum LTin agent count
 */
extern int get_agent_LTin_MAX_count();



/** get_agent_LTin_ltin_random_movement_count
 * Gets the agent count for the LTin agent type in state ltin_random_movement
 * @return		the current LTin agent count in state ltin_random_movement
 */
extern int get_agent_LTin_ltin_random_movement_count();

/** reset_ltin_random_movement_count
 * Resets the agent count of the LTin in state ltin_random_movement to 0. This is useful for interacting with some visualisations.
 */
extern void reset_LTin_ltin_random_movement_count();

/** get_device_LTin_ltin_random_movement_agents
 * Gets a pointer to xmachine_memory_LTin_list on the GPU device
 * @return		a xmachine_memory_LTin_list on the GPU device
 */
extern xmachine_memory_LTin_list* get_device_LTin_ltin_random_movement_agents();

/** get_host_LTin_ltin_random_movement_agents
 * Gets a pointer to xmachine_memory_LTin_list on the CPU host
 * @return		a xmachine_memory_LTin_list on the CPU host
 */
extern xmachine_memory_LTin_list* get_host_LTin_ltin_random_movement_agents();


/** sort_LTins_ltin_random_movement
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_LTins_ltin_random_movement(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTin_list* agents));


/** get_agent_LTin_stable_contact_count
 * Gets the agent count for the LTin agent type in state stable_contact
 * @return		the current LTin agent count in state stable_contact
 */
extern int get_agent_LTin_stable_contact_count();

/** reset_stable_contact_count
 * Resets the agent count of the LTin in state stable_contact to 0. This is useful for interacting with some visualisations.
 */
extern void reset_LTin_stable_contact_count();

/** get_device_LTin_stable_contact_agents
 * Gets a pointer to xmachine_memory_LTin_list on the GPU device
 * @return		a xmachine_memory_LTin_list on the GPU device
 */
extern xmachine_memory_LTin_list* get_device_LTin_stable_contact_agents();

/** get_host_LTin_stable_contact_agents
 * Gets a pointer to xmachine_memory_LTin_list on the CPU host
 * @return		a xmachine_memory_LTin_list on the CPU host
 */
extern xmachine_memory_LTin_list* get_host_LTin_stable_contact_agents();


/** sort_LTins_stable_contact
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_LTins_stable_contact(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTin_list* agents));


/** get_agent_LTin_localised_movement_count
 * Gets the agent count for the LTin agent type in state localised_movement
 * @return		the current LTin agent count in state localised_movement
 */
extern int get_agent_LTin_localised_movement_count();

/** reset_localised_movement_count
 * Resets the agent count of the LTin in state localised_movement to 0. This is useful for interacting with some visualisations.
 */
extern void reset_LTin_localised_movement_count();

/** get_device_LTin_localised_movement_agents
 * Gets a pointer to xmachine_memory_LTin_list on the GPU device
 * @return		a xmachine_memory_LTin_list on the GPU device
 */
extern xmachine_memory_LTin_list* get_device_LTin_localised_movement_agents();

/** get_host_LTin_localised_movement_agents
 * Gets a pointer to xmachine_memory_LTin_list on the CPU host
 * @return		a xmachine_memory_LTin_list on the CPU host
 */
extern xmachine_memory_LTin_list* get_host_LTin_localised_movement_agents();


/** sort_LTins_localised_movement
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_LTins_localised_movement(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTin_list* agents));


    
/** get_agent_LTi_MAX_count
 * Gets the max agent count for the LTi agent type 
 * @return		the maximum LTi agent count
 */
extern int get_agent_LTi_MAX_count();



/** get_agent_LTi_lti_random_movement_count
 * Gets the agent count for the LTi agent type in state lti_random_movement
 * @return		the current LTi agent count in state lti_random_movement
 */
extern int get_agent_LTi_lti_random_movement_count();

/** reset_lti_random_movement_count
 * Resets the agent count of the LTi in state lti_random_movement to 0. This is useful for interacting with some visualisations.
 */
extern void reset_LTi_lti_random_movement_count();

/** get_device_LTi_lti_random_movement_agents
 * Gets a pointer to xmachine_memory_LTi_list on the GPU device
 * @return		a xmachine_memory_LTi_list on the GPU device
 */
extern xmachine_memory_LTi_list* get_device_LTi_lti_random_movement_agents();

/** get_host_LTi_lti_random_movement_agents
 * Gets a pointer to xmachine_memory_LTi_list on the CPU host
 * @return		a xmachine_memory_LTi_list on the CPU host
 */
extern xmachine_memory_LTi_list* get_host_LTi_lti_random_movement_agents();


/** sort_LTis_lti_random_movement
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_LTis_lti_random_movement(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTi_list* agents));


/** get_agent_LTi_responding_count
 * Gets the agent count for the LTi agent type in state responding
 * @return		the current LTi agent count in state responding
 */
extern int get_agent_LTi_responding_count();

/** reset_responding_count
 * Resets the agent count of the LTi in state responding to 0. This is useful for interacting with some visualisations.
 */
extern void reset_LTi_responding_count();

/** get_device_LTi_responding_agents
 * Gets a pointer to xmachine_memory_LTi_list on the GPU device
 * @return		a xmachine_memory_LTi_list on the GPU device
 */
extern xmachine_memory_LTi_list* get_device_LTi_responding_agents();

/** get_host_LTi_responding_agents
 * Gets a pointer to xmachine_memory_LTi_list on the CPU host
 * @return		a xmachine_memory_LTi_list on the CPU host
 */
extern xmachine_memory_LTi_list* get_host_LTi_responding_agents();


/** sort_LTis_responding
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_LTis_responding(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTi_list* agents));


/** get_agent_LTi_contact_count
 * Gets the agent count for the LTi agent type in state contact
 * @return		the current LTi agent count in state contact
 */
extern int get_agent_LTi_contact_count();

/** reset_contact_count
 * Resets the agent count of the LTi in state contact to 0. This is useful for interacting with some visualisations.
 */
extern void reset_LTi_contact_count();

/** get_device_LTi_contact_agents
 * Gets a pointer to xmachine_memory_LTi_list on the GPU device
 * @return		a xmachine_memory_LTi_list on the GPU device
 */
extern xmachine_memory_LTi_list* get_device_LTi_contact_agents();

/** get_host_LTi_contact_agents
 * Gets a pointer to xmachine_memory_LTi_list on the CPU host
 * @return		a xmachine_memory_LTi_list on the CPU host
 */
extern xmachine_memory_LTi_list* get_host_LTi_contact_agents();


/** sort_LTis_contact
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_LTis_contact(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTi_list* agents));


/** get_agent_LTi_adhesion_count
 * Gets the agent count for the LTi agent type in state adhesion
 * @return		the current LTi agent count in state adhesion
 */
extern int get_agent_LTi_adhesion_count();

/** reset_adhesion_count
 * Resets the agent count of the LTi in state adhesion to 0. This is useful for interacting with some visualisations.
 */
extern void reset_LTi_adhesion_count();

/** get_device_LTi_adhesion_agents
 * Gets a pointer to xmachine_memory_LTi_list on the GPU device
 * @return		a xmachine_memory_LTi_list on the GPU device
 */
extern xmachine_memory_LTi_list* get_device_LTi_adhesion_agents();

/** get_host_LTi_adhesion_agents
 * Gets a pointer to xmachine_memory_LTi_list on the CPU host
 * @return		a xmachine_memory_LTi_list on the CPU host
 */
extern xmachine_memory_LTi_list* get_host_LTi_adhesion_agents();


/** sort_LTis_adhesion
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_LTis_adhesion(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTi_list* agents));


    
/** get_agent_LTo_MAX_count
 * Gets the max agent count for the LTo agent type 
 * @return		the maximum LTo agent count
 */
extern int get_agent_LTo_MAX_count();



/** get_agent_LTo_no_expression_count
 * Gets the agent count for the LTo agent type in state no_expression
 * @return		the current LTo agent count in state no_expression
 */
extern int get_agent_LTo_no_expression_count();

/** reset_no_expression_count
 * Resets the agent count of the LTo in state no_expression to 0. This is useful for interacting with some visualisations.
 */
extern void reset_LTo_no_expression_count();

/** get_device_LTo_no_expression_agents
 * Gets a pointer to xmachine_memory_LTo_list on the GPU device
 * @return		a xmachine_memory_LTo_list on the GPU device
 */
extern xmachine_memory_LTo_list* get_device_LTo_no_expression_agents();

/** get_host_LTo_no_expression_agents
 * Gets a pointer to xmachine_memory_LTo_list on the CPU host
 * @return		a xmachine_memory_LTo_list on the CPU host
 */
extern xmachine_memory_LTo_list* get_host_LTo_no_expression_agents();


/** sort_LTos_no_expression
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_LTos_no_expression(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTo_list* agents));


/** get_agent_LTo_expression_count
 * Gets the agent count for the LTo agent type in state expression
 * @return		the current LTo agent count in state expression
 */
extern int get_agent_LTo_expression_count();

/** reset_expression_count
 * Resets the agent count of the LTo in state expression to 0. This is useful for interacting with some visualisations.
 */
extern void reset_LTo_expression_count();

/** get_device_LTo_expression_agents
 * Gets a pointer to xmachine_memory_LTo_list on the GPU device
 * @return		a xmachine_memory_LTo_list on the GPU device
 */
extern xmachine_memory_LTo_list* get_device_LTo_expression_agents();

/** get_host_LTo_expression_agents
 * Gets a pointer to xmachine_memory_LTo_list on the CPU host
 * @return		a xmachine_memory_LTo_list on the CPU host
 */
extern xmachine_memory_LTo_list* get_host_LTo_expression_agents();


/** sort_LTos_expression
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_LTos_expression(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTo_list* agents));


/** get_agent_LTo_adhesion_upregulation_count
 * Gets the agent count for the LTo agent type in state adhesion_upregulation
 * @return		the current LTo agent count in state adhesion_upregulation
 */
extern int get_agent_LTo_adhesion_upregulation_count();

/** reset_adhesion_upregulation_count
 * Resets the agent count of the LTo in state adhesion_upregulation to 0. This is useful for interacting with some visualisations.
 */
extern void reset_LTo_adhesion_upregulation_count();

/** get_device_LTo_adhesion_upregulation_agents
 * Gets a pointer to xmachine_memory_LTo_list on the GPU device
 * @return		a xmachine_memory_LTo_list on the GPU device
 */
extern xmachine_memory_LTo_list* get_device_LTo_adhesion_upregulation_agents();

/** get_host_LTo_adhesion_upregulation_agents
 * Gets a pointer to xmachine_memory_LTo_list on the CPU host
 * @return		a xmachine_memory_LTo_list on the CPU host
 */
extern xmachine_memory_LTo_list* get_host_LTo_adhesion_upregulation_agents();


/** sort_LTos_adhesion_upregulation
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_LTos_adhesion_upregulation(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTo_list* agents));


/** get_agent_LTo_chemokine_upregulation_count
 * Gets the agent count for the LTo agent type in state chemokine_upregulation
 * @return		the current LTo agent count in state chemokine_upregulation
 */
extern int get_agent_LTo_chemokine_upregulation_count();

/** reset_chemokine_upregulation_count
 * Resets the agent count of the LTo in state chemokine_upregulation to 0. This is useful for interacting with some visualisations.
 */
extern void reset_LTo_chemokine_upregulation_count();

/** get_device_LTo_chemokine_upregulation_agents
 * Gets a pointer to xmachine_memory_LTo_list on the GPU device
 * @return		a xmachine_memory_LTo_list on the GPU device
 */
extern xmachine_memory_LTo_list* get_device_LTo_chemokine_upregulation_agents();

/** get_host_LTo_chemokine_upregulation_agents
 * Gets a pointer to xmachine_memory_LTo_list on the CPU host
 * @return		a xmachine_memory_LTo_list on the CPU host
 */
extern xmachine_memory_LTo_list* get_host_LTo_chemokine_upregulation_agents();


/** sort_LTos_chemokine_upregulation
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_LTos_chemokine_upregulation(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTo_list* agents));


/** get_agent_LTo_mature_count
 * Gets the agent count for the LTo agent type in state mature
 * @return		the current LTo agent count in state mature
 */
extern int get_agent_LTo_mature_count();

/** reset_mature_count
 * Resets the agent count of the LTo in state mature to 0. This is useful for interacting with some visualisations.
 */
extern void reset_LTo_mature_count();

/** get_device_LTo_mature_agents
 * Gets a pointer to xmachine_memory_LTo_list on the GPU device
 * @return		a xmachine_memory_LTo_list on the GPU device
 */
extern xmachine_memory_LTo_list* get_device_LTo_mature_agents();

/** get_host_LTo_mature_agents
 * Gets a pointer to xmachine_memory_LTo_list on the CPU host
 * @return		a xmachine_memory_LTo_list on the CPU host
 */
extern xmachine_memory_LTo_list* get_host_LTo_mature_agents();


/** sort_LTos_mature
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_LTos_mature(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTo_list* agents));


/** get_agent_LTo_downregulated_count
 * Gets the agent count for the LTo agent type in state downregulated
 * @return		the current LTo agent count in state downregulated
 */
extern int get_agent_LTo_downregulated_count();

/** reset_downregulated_count
 * Resets the agent count of the LTo in state downregulated to 0. This is useful for interacting with some visualisations.
 */
extern void reset_LTo_downregulated_count();

/** get_device_LTo_downregulated_agents
 * Gets a pointer to xmachine_memory_LTo_list on the GPU device
 * @return		a xmachine_memory_LTo_list on the GPU device
 */
extern xmachine_memory_LTo_list* get_device_LTo_downregulated_agents();

/** get_host_LTo_downregulated_agents
 * Gets a pointer to xmachine_memory_LTo_list on the CPU host
 * @return		a xmachine_memory_LTo_list on the CPU host
 */
extern xmachine_memory_LTo_list* get_host_LTo_downregulated_agents();


/** sort_LTos_downregulated
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_LTos_downregulated(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_LTo_list* agents));


  
  
/* Analytics functions for each varible in each state*/
typedef enum {
  REDUCTION_MAX,
  REDUCTION_MIN,
  REDUCTION_SUM
}reduction_operator;


/** float reduce_LTin_ltin_random_movement_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTin_ltin_random_movement_x_variable();



/** float reduce_LTin_ltin_random_movement_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTin_ltin_random_movement_y_variable();



/** float reduce_LTin_stable_contact_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTin_stable_contact_x_variable();



/** float reduce_LTin_stable_contact_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTin_stable_contact_y_variable();



/** float reduce_LTin_localised_movement_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTin_localised_movement_x_variable();



/** float reduce_LTin_localised_movement_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTin_localised_movement_y_variable();



/** float reduce_LTi_lti_random_movement_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTi_lti_random_movement_x_variable();



/** float reduce_LTi_lti_random_movement_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTi_lti_random_movement_y_variable();



/** float reduce_LTi_responding_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTi_responding_x_variable();



/** float reduce_LTi_responding_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTi_responding_y_variable();



/** float reduce_LTi_contact_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTi_contact_x_variable();



/** float reduce_LTi_contact_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTi_contact_y_variable();



/** float reduce_LTi_adhesion_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTi_adhesion_x_variable();



/** float reduce_LTi_adhesion_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTi_adhesion_y_variable();



/** float reduce_LTo_no_expression_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTo_no_expression_x_variable();



/** float reduce_LTo_no_expression_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTo_no_expression_y_variable();



/** float reduce_LTo_expression_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTo_expression_x_variable();



/** float reduce_LTo_expression_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTo_expression_y_variable();



/** float reduce_LTo_adhesion_upregulation_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTo_adhesion_upregulation_x_variable();



/** float reduce_LTo_adhesion_upregulation_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTo_adhesion_upregulation_y_variable();



/** float reduce_LTo_chemokine_upregulation_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTo_chemokine_upregulation_x_variable();



/** float reduce_LTo_chemokine_upregulation_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTo_chemokine_upregulation_y_variable();



/** float reduce_LTo_mature_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTo_mature_x_variable();



/** float reduce_LTo_mature_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTo_mature_y_variable();



/** float reduce_LTo_downregulated_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTo_downregulated_x_variable();



/** float reduce_LTo_downregulated_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_LTo_downregulated_y_variable();




  
/* global constant variables */

__constant__ int LTI_CELL_SIZE;

__constant__ int LTO_CELL_SIZE;

__constant__ int adhesion_distance_threshold;

__constant__ float CHEMO_THRESHOLD;

__constant__ int CHEMO_CURVE_ADJUST;

__constant__ float CHEMO_UPPER_ADJUST;

__constant__ float CHEMO_LOWER_ADJUST;

__constant__ float INCREASE_CHEMO_EXPRESSION;

__constant__ float INITIAL_ADHESION;

__constant__ float ADHESION_SLOPE;

__constant__ float ADHESION_INCREMENT;

__constant__ float MAX_ADHESION_PROBABILITY;

__constant__ int INITIAL_CIRCUMFERENCE;

__constant__ int MAXIMUM_CIRCUMFERENCE;

__constant__ int INITIAL_LENGTH;

__constant__ int MAXIMUM_LENGTH;

__constant__ float STROMAL_CELL_DENSITY;

__constant__ int GROWTH_TIME;

__constant__ float PERCENT_LTIN_FROM_FACS;

__constant__ float PERCENT_LTI_FROM_FACS;

/** set_LTI_CELL_SIZE
 * Sets the constant variable LTI_CELL_SIZE on the device which can then be used in the agent functions.
 * @param h_LTI_CELL_SIZE value to set the variable
 */
extern void set_LTI_CELL_SIZE(int* h_LTI_CELL_SIZE);


extern const int* get_LTI_CELL_SIZE();


extern int h_env_LTI_CELL_SIZE;

/** set_LTO_CELL_SIZE
 * Sets the constant variable LTO_CELL_SIZE on the device which can then be used in the agent functions.
 * @param h_LTO_CELL_SIZE value to set the variable
 */
extern void set_LTO_CELL_SIZE(int* h_LTO_CELL_SIZE);


extern const int* get_LTO_CELL_SIZE();


extern int h_env_LTO_CELL_SIZE;

/** set_adhesion_distance_threshold
 * Sets the constant variable adhesion_distance_threshold on the device which can then be used in the agent functions.
 * @param h_adhesion_distance_threshold value to set the variable
 */
extern void set_adhesion_distance_threshold(int* h_adhesion_distance_threshold);


extern const int* get_adhesion_distance_threshold();


extern int h_env_adhesion_distance_threshold;

/** set_CHEMO_THRESHOLD
 * Sets the constant variable CHEMO_THRESHOLD on the device which can then be used in the agent functions.
 * @param h_CHEMO_THRESHOLD value to set the variable
 */
extern void set_CHEMO_THRESHOLD(float* h_CHEMO_THRESHOLD);


extern const float* get_CHEMO_THRESHOLD();


extern float h_env_CHEMO_THRESHOLD;

/** set_CHEMO_CURVE_ADJUST
 * Sets the constant variable CHEMO_CURVE_ADJUST on the device which can then be used in the agent functions.
 * @param h_CHEMO_CURVE_ADJUST value to set the variable
 */
extern void set_CHEMO_CURVE_ADJUST(int* h_CHEMO_CURVE_ADJUST);


extern const int* get_CHEMO_CURVE_ADJUST();


extern int h_env_CHEMO_CURVE_ADJUST;

/** set_CHEMO_UPPER_ADJUST
 * Sets the constant variable CHEMO_UPPER_ADJUST on the device which can then be used in the agent functions.
 * @param h_CHEMO_UPPER_ADJUST value to set the variable
 */
extern void set_CHEMO_UPPER_ADJUST(float* h_CHEMO_UPPER_ADJUST);


extern const float* get_CHEMO_UPPER_ADJUST();


extern float h_env_CHEMO_UPPER_ADJUST;

/** set_CHEMO_LOWER_ADJUST
 * Sets the constant variable CHEMO_LOWER_ADJUST on the device which can then be used in the agent functions.
 * @param h_CHEMO_LOWER_ADJUST value to set the variable
 */
extern void set_CHEMO_LOWER_ADJUST(float* h_CHEMO_LOWER_ADJUST);


extern const float* get_CHEMO_LOWER_ADJUST();


extern float h_env_CHEMO_LOWER_ADJUST;

/** set_INCREASE_CHEMO_EXPRESSION
 * Sets the constant variable INCREASE_CHEMO_EXPRESSION on the device which can then be used in the agent functions.
 * @param h_INCREASE_CHEMO_EXPRESSION value to set the variable
 */
extern void set_INCREASE_CHEMO_EXPRESSION(float* h_INCREASE_CHEMO_EXPRESSION);


extern const float* get_INCREASE_CHEMO_EXPRESSION();


extern float h_env_INCREASE_CHEMO_EXPRESSION;

/** set_INITIAL_ADHESION
 * Sets the constant variable INITIAL_ADHESION on the device which can then be used in the agent functions.
 * @param h_INITIAL_ADHESION value to set the variable
 */
extern void set_INITIAL_ADHESION(float* h_INITIAL_ADHESION);


extern const float* get_INITIAL_ADHESION();


extern float h_env_INITIAL_ADHESION;

/** set_ADHESION_SLOPE
 * Sets the constant variable ADHESION_SLOPE on the device which can then be used in the agent functions.
 * @param h_ADHESION_SLOPE value to set the variable
 */
extern void set_ADHESION_SLOPE(float* h_ADHESION_SLOPE);


extern const float* get_ADHESION_SLOPE();


extern float h_env_ADHESION_SLOPE;

/** set_ADHESION_INCREMENT
 * Sets the constant variable ADHESION_INCREMENT on the device which can then be used in the agent functions.
 * @param h_ADHESION_INCREMENT value to set the variable
 */
extern void set_ADHESION_INCREMENT(float* h_ADHESION_INCREMENT);


extern const float* get_ADHESION_INCREMENT();


extern float h_env_ADHESION_INCREMENT;

/** set_MAX_ADHESION_PROBABILITY
 * Sets the constant variable MAX_ADHESION_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_MAX_ADHESION_PROBABILITY value to set the variable
 */
extern void set_MAX_ADHESION_PROBABILITY(float* h_MAX_ADHESION_PROBABILITY);


extern const float* get_MAX_ADHESION_PROBABILITY();


extern float h_env_MAX_ADHESION_PROBABILITY;

/** set_INITIAL_CIRCUMFERENCE
 * Sets the constant variable INITIAL_CIRCUMFERENCE on the device which can then be used in the agent functions.
 * @param h_INITIAL_CIRCUMFERENCE value to set the variable
 */
extern void set_INITIAL_CIRCUMFERENCE(int* h_INITIAL_CIRCUMFERENCE);


extern const int* get_INITIAL_CIRCUMFERENCE();


extern int h_env_INITIAL_CIRCUMFERENCE;

/** set_MAXIMUM_CIRCUMFERENCE
 * Sets the constant variable MAXIMUM_CIRCUMFERENCE on the device which can then be used in the agent functions.
 * @param h_MAXIMUM_CIRCUMFERENCE value to set the variable
 */
extern void set_MAXIMUM_CIRCUMFERENCE(int* h_MAXIMUM_CIRCUMFERENCE);


extern const int* get_MAXIMUM_CIRCUMFERENCE();


extern int h_env_MAXIMUM_CIRCUMFERENCE;

/** set_INITIAL_LENGTH
 * Sets the constant variable INITIAL_LENGTH on the device which can then be used in the agent functions.
 * @param h_INITIAL_LENGTH value to set the variable
 */
extern void set_INITIAL_LENGTH(int* h_INITIAL_LENGTH);


extern const int* get_INITIAL_LENGTH();


extern int h_env_INITIAL_LENGTH;

/** set_MAXIMUM_LENGTH
 * Sets the constant variable MAXIMUM_LENGTH on the device which can then be used in the agent functions.
 * @param h_MAXIMUM_LENGTH value to set the variable
 */
extern void set_MAXIMUM_LENGTH(int* h_MAXIMUM_LENGTH);


extern const int* get_MAXIMUM_LENGTH();


extern int h_env_MAXIMUM_LENGTH;

/** set_STROMAL_CELL_DENSITY
 * Sets the constant variable STROMAL_CELL_DENSITY on the device which can then be used in the agent functions.
 * @param h_STROMAL_CELL_DENSITY value to set the variable
 */
extern void set_STROMAL_CELL_DENSITY(float* h_STROMAL_CELL_DENSITY);


extern const float* get_STROMAL_CELL_DENSITY();


extern float h_env_STROMAL_CELL_DENSITY;

/** set_GROWTH_TIME
 * Sets the constant variable GROWTH_TIME on the device which can then be used in the agent functions.
 * @param h_GROWTH_TIME value to set the variable
 */
extern void set_GROWTH_TIME(int* h_GROWTH_TIME);


extern const int* get_GROWTH_TIME();


extern int h_env_GROWTH_TIME;

/** set_PERCENT_LTIN_FROM_FACS
 * Sets the constant variable PERCENT_LTIN_FROM_FACS on the device which can then be used in the agent functions.
 * @param h_PERCENT_LTIN_FROM_FACS value to set the variable
 */
extern void set_PERCENT_LTIN_FROM_FACS(float* h_PERCENT_LTIN_FROM_FACS);


extern const float* get_PERCENT_LTIN_FROM_FACS();


extern float h_env_PERCENT_LTIN_FROM_FACS;

/** set_PERCENT_LTI_FROM_FACS
 * Sets the constant variable PERCENT_LTI_FROM_FACS on the device which can then be used in the agent functions.
 * @param h_PERCENT_LTI_FROM_FACS value to set the variable
 */
extern void set_PERCENT_LTI_FROM_FACS(float* h_PERCENT_LTI_FROM_FACS);


extern const float* get_PERCENT_LTI_FROM_FACS();


extern float h_env_PERCENT_LTI_FROM_FACS;


/** getMaximumBound
 * Returns the maximum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the maximum x, y and z positions of all agents
 */
glm::vec3 getMaximumBounds();

/** getMinimumBounds
 * Returns the minimum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the minimum x, y and z positions of all agents
 */
glm::vec3 getMinimumBounds();
    
    
#ifdef VISUALISATION
/** initVisualisation
 * Prototype for method which initialises the visualisation. Must be implemented in separate file
 * @param argc	the argument count from the main function used with GLUT
 * @param argv	the argument values from the main function used with GLUT
 */
extern void initVisualisation();

extern void runVisualisation();


#endif

#endif //__HEADER

