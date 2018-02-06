
/*
 * Copyright 2018 University of York.
 * Author: Oliver Binns
 * Contact: ob601@york.ac.uk (mail@oliverbinns.co.uk)
 *
 * 
 */

#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include <header.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>

inline __device__ float dot(glm::vec2 a, glm::vec2 b)
{
    return a.x * b.x + a.y * b.y;
}

inline __device__ float length(glm::vec2 v)
{
    return sqrtf(dot(v, v));
}

inline __device__ float adhesionProbability(){
    float value = 0;
    return (value > MAX_ADHESION_PROBABILITY) ? value : MAX_ADHESION_PROBABILITY;
}

inline __device__ float chemokineLevel(float distanceToLTo){
    float value = -(-CHEMO_THRESHOLD + distanceToLTo + CHEMO_THRESHOLD);
    value = expf(value);
    value = 1 + value;
    return 1 / value;
}

/**
 * Ensure that the agent is within the boundaries of the gut area.
 *
 * TODO: If it goes outside of y direction, the agent should die as it has migrated
 * If it goes outside the x direction, it wraps around 
 */
__FLAME_GPU_FUNC__ glm::vec2 boundPosition(glm::vec2 agent_position){
    //Wrap around to min/max values if OUTSIDE range
    //TODO, fix this to move correct number of coordinates, rather than to exact min/max pos
    agent_position.x = (agent_position.x < 0) ? MAXIMUM_LENGTH: agent_position.x;
    agent_position.x = (agent_position.x > MAXIMUM_LENGTH) ? 0: agent_position.x;
    
    
    //TODO: AGENT SHOULD DIE if it goes outside Y direction
    //agent_position.y = (agent_position.y < 0)? MAX_POSITION: agent_position.y;
    //agent_position.y = (agent_position.y > MAX_POSITION)? MIN_POSITION: agent_position.y;

    return agent_position;
}

/*
 * Movement function, calculates the new position for an agent travelling at the given velocity in a random direction.
 */
__FLAME_GPU_FUNC__ glm::vec2 random_move(glm::vec2 position, float velocity, RNG_rand48* rand48)
{
	//Calculate velocity
	float angle = 2 * M_PI *  rnd<CONTINUOUS>(rand48);

	float x_move = velocity * sinf(angle);
	float y_move = velocity * cosf(angle);
	glm::vec2 agent_velocity = glm::vec2(x_move, y_move);

	position += agent_velocity;
	
	return boundPosition(position);
}

/**
 * random_move FLAMEGPU Agent Function
 * This method is responsible for the agent moving randomly around the plane.
 *
 * @param agent Pointer to an LTi agent. This represents a single agent instance and can be modified directly.
 * @param location_messages Pointer to output message list of type xmachine_message_example_message_list.
 * 
 * @return Return type is always int. 0 means the agent does NOT die.
 */
__FLAME_GPU_FUNC__ int lti_random_move(xmachine_memory_LTi* xmemory,
    RNG_rand48* rand48)
{
    glm::vec2 init_position = glm::vec2(xmemory->x, xmemory->y);
    glm::vec2 new_position = random_move(init_position, xmemory->velocity, rand48);
    
    xmemory->x = new_position.x;
    xmemory->y = new_position.y;
    
    return 0;
}

__FLAME_GPU_FUNC__ int ltin_random_move(xmachine_memory_LTin* xmemory,
    RNG_rand48* rand48)
{
    glm::vec2 init_position = glm::vec2(xmemory->x, xmemory->y);
    glm::vec2 new_position = random_move(init_position, xmemory->velocity, rand48);
    
    xmemory->x = new_position.x;
    xmemory->y = new_position.y;
    
    return 0;
}

__FLAME_GPU_FUNC__ int express(xmachine_memory_LTo* xmemory
							   xmachine_message_location_list* location_messages)
{
	add_location_message(location_messages, LTO_AGENT_TYPE, xmemory->x, xmemory->y)
}

#endif //_FLAMEGPU_FUNCTIONS

