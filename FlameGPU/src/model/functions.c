
/*
 * Copyright 2018 University of York.
 * Author: Oliver Binns
 * Contact: ob601@york.ac.uk (mail@oliverbinns.co.uk)
 *
 * 
 */

#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include "header.h"

__FLAME_GPU_INIT_FUNC__ void setConstants(){
	int LTI_AGENT_TYPE = 0;
	set_LTI_AGENT_TYPE(&LTI_AGENT_TYPE);
	int LTIN_AGENT_TYPE = 1;
	set_LTIN_AGENT_TYPE(&LTIN_AGENT_TYPE);
	int LTO_AGENT_TYPE = 2;
	set_LTO_AGENT_TYPE(&LTO_AGENT_TYPE);

	int LTI_CELL_SIZE = 6;
	set_LTI_CELL_SIZE(&LTI_CELL_SIZE);
	int LTO_CELL_SIZE = 2;
	set_LTO_CELL_SIZE(&LTO_CELL_SIZE);

	//int ADHESION_DISTANCE_THRESHOLD = (LTI_CELL_SIZE + LTO_CELL_SIZE) / 2;
	//set_ADHESION_DISTANCE_THRESHOLD(&ADHESION_DISTANCE_THRESHOLD);

	//Modelling Chemokines
	float CHEMO_THRESHOLD = 0.3f;
	set_CHEMO_THRESHOLD(&CHEMO_THRESHOLD);
	int CHEMO_CURVE_ADJUST = 3;
	set_CHEMO_CURVE_ADJUST(&CHEMO_CURVE_ADJUST);
	float CHEMO_LOWER_ADJUST = 0.04f;
	set_CHEMO_LOWER_ADJUST(&CHEMO_LOWER_ADJUST);
	float CHEMO_UPPER_ADJUST = 0.2f;
	set_CHEMO_UPPER_ADJUST(&CHEMO_UPPER_ADJUST);
	float INCREASE_CHEMO_EXPRESSION = 0.005f;
	set_INCREASE_CHEMO_EXPRESSION(&INCREASE_CHEMO_EXPRESSION);

	//Modelling Adhesion
	float INITIAL_ADHESION = 0;
	set_INITIAL_ADHESION(&INITIAL_ADHESION);
	float ADHESION_SLOPE = 1;
	set_ADHESION_SLOPE(&ADHESION_SLOPE);
	float ADHESION_INCREMENT = 0.05f;
	set_ADHESION_INCREMENT(&ADHESION_INCREMENT);
	float MAX_ADHESION_PROBABILITY = 0.65f;
	set_MAX_ADHESION_PROBABILITY(&MAX_ADHESION_PROBABILITY);

	int MAX_CELL_SPEED = 10;
	set_MAX_CELL_SPEED(&MAX_CELL_SPEED);
	int INITIAL_CIRCUMFERENCE = 244;
	set_INITIAL_CIRCUMFERENCE(&INITIAL_CIRCUMFERENCE);
	int MAXIMUM_CIRCUMFERENCE = 254;
	set_MAXIMUM_CIRCUMFERENCE(&MAXIMUM_CIRCUMFERENCE);
	int INITIAL_LENGTH = 7203;
	set_INITIAL_LENGTH(&INITIAL_LENGTH);
	int MAXIMUM_LENGTH = 7303;
	set_MAXIMUM_LENGTH(&MAXIMUM_LENGTH);
	float STROMAL_CELL_DENSITY = 0.2f;
	set_STROMAL_CELL_DENSITY(&STROMAL_CELL_DENSITY);
	int GROWTH_TIME = 72;
	set_GROWTH_TIME(&GROWTH_TIME);
	float PERCENT_LTIN_FROM_FACS = 0.45f;
	set_PERCENT_LTIN_FROM_FACS(&PERCENT_LTIN_FROM_FACS);
	float PERCENT_LTI_FROM_FACS = 0.37;
	set_PERCENT_LTI_FROM_FACS(&PERCENT_LTI_FROM_FACS);	
}

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
    agent_position.x = fmod(agent_position.x, float(MAXIMUM_LENGTH));
    
    //TODO: AGENT SHOULD DIE if it goes outside Y direction
    agent_position.y = fmod(agent_position.y, float(MAXIMUM_CIRCUMFERENCE));

    return agent_position;
}

/*
 * Movement function, calculates the new position for an agent travelling at the given velocity in a random direction.
 */
__FLAME_GPU_FUNC__ glm::vec2 random_move(glm::vec2 position,
                                         float velocity,
                                         RNG_rand48* rand48)
{
	//Calculate velocity
	float angle = 2 * M_PI *  rnd<CONTINUOUS>(rand48);

	float x_move = 10 * velocity * sinf(angle);
	float y_move = 10 * velocity * cosf(angle);

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

__FLAME_GPU_FUNC__ int express(xmachine_memory_LTo* xmemory,
							   xmachine_message_location_list* location_messages)
{
	int x = xmemory->x;
	int y = xmemory->y;
	add_location_message(location_messages, x, y, 0, 0);
    
    return 0;
}

#endif //_FLAMEGPU_FUNCTIONS