
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
    int SIM_STEP = 1;
    set_SIM_STEP(&SIM_STEP);

	int LTI_AGENT_TYPE = 1;
	set_LTI_AGENT_TYPE(&LTI_AGENT_TYPE);
	int LTIN_AGENT_TYPE = 7;
	set_LTIN_AGENT_TYPE(&LTIN_AGENT_TYPE);
	int LTO_AGENT_TYPE = 0;
	set_LTO_AGENT_TYPE(&LTO_AGENT_TYPE);

	int LTI_CELL_SIZE = 6;
	set_LTI_CELL_SIZE(&LTI_CELL_SIZE);
	int LTO_CELL_SIZE = 2;
	set_LTO_CELL_SIZE(&LTO_CELL_SIZE);

	int ADHESION_DISTANCE_THRESHOLD = (LTI_CELL_SIZE + LTO_CELL_SIZE) / 2;
	set_ADHESION_DISTANCE_THRESHOLD(&ADHESION_DISTANCE_THRESHOLD);

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
	int CIRCUMFERENCE = 254;
	set_CIRCUMFERENCE(&CIRCUMFERENCE);
	int LENGTH = 7303;
	set_LENGTH(&LENGTH);
	float STROMAL_CELL_DENSITY = 0.2f;
	set_STROMAL_CELL_DENSITY(&STROMAL_CELL_DENSITY);
	float PERCENT_LTIN_FROM_FACS = 0.0045f;
	set_PERCENT_LTIN_FROM_FACS(&PERCENT_LTIN_FROM_FACS);
	float PERCENT_LTI_FROM_FACS = 0.0037;
	set_PERCENT_LTI_FROM_FACS(&PERCENT_LTI_FROM_FACS);	
}

__device__ float adhesionProbability(){
    float value = 0;
    return (value > MAX_ADHESION_PROBABILITY) ? value : MAX_ADHESION_PROBABILITY;
}

__device__ float chemokineLevel(float distanceToLTo, float lto_expression_level){
    float value = -(-lto_expression_level * distanceToLTo + CHEMO_CURVE_ADJUST);
    value = expf(value);
    value = 1 + value;
    return 1 / value;
}

/*
 * This method returns a random value within 0-1
 */
float randomUniform(){
	return (float)rand() / (float)RAND_MAX;
}

/*
 * This method generates a random value from the Gaussian distribution
 * using the given mean and standard deviation.
 */
float genGaussian(float mean, float std){
    //Uses the Polar method to generate Gaussian values
    float r1, r2, w, mult;
    //Two values are created, so one is stored in a static variable until this function is next called.
    static float x1, x2;
    //Call is a boolean flag to determine whether this is our initial call.
    static int call = 0;
    
    if(call == 1){
        call = !call;
        return (mean + std * x2);
    }
    do{
        r1 = randomUniform();
        r2 = randomUniform();
        w = powf(r1, 2) + powf(r2, 2);
    }while(w >= 1 || w == 0);
    
    mult = sqrtf((-2 * logf(w)) / w);
    x1 = r1 * mult;
    x2 = r2 * mult;
    
    call = !call;
    
    return mean + std * x1;
}

/*
 * TODO: truncate the value to ensure it is within the range 0, 1
 */
float randomGaussian(){
    return genGaussian(0.5, 0.2);
}

__FLAME_GPU_STEP_FUNC__ void migrateNewCells(){
    //INCREMENT SIM STEP:
    int step = *get_SIM_STEP() + 1;
    set_SIM_STEP(&step);

    //TODO: refactor LTi, LTin migration to share code, if possible:
	//CREATE LTis:
	// Can create upto h_agent_AoS_MAX agents in a single pass (the number allocated for) but the full amount does not have to be created.
    const float spaces = (float)(*get_LENGTH() * *get_CIRCUMFERENCE() / *get_LTI_CELL_SIZE());
    const float steps_24h = 24.0f * 60.0f;

	const float lti_migration_rate = spaces * *get_PERCENT_LTI_FROM_FACS() / steps_24h;
    unsigned int lti_to_add = 0;
    if(lti_migration_rate > 1){
        lti_to_add = floor(lti_migration_rate);
    }else if((step % 100) > (lti_migration_rate * 100)){
        //^ modulo with 100 for Percentage:
        lti_to_add = 1;
    }
    
    // It is sensible to check if it is possible to create new agents, and if so how many.
	unsigned int agent_remaining = get_agent_LTi_MAX_count() - get_agent_LTi_lti_random_movement_count();
	if (agent_remaining > 0) {
		unsigned int count = (lti_to_add > agent_remaining) ? agent_remaining: lti_to_add;
		xmachine_memory_LTi** agents = h_allocate_agent_LTi_array(count);
        // Populate data as required
		for (unsigned int i = 0; i < count; i++) {
			//agents[i] = h_allocate_agent_LTi();
			//Initialise agent variables:
			agents[i]->x = randomUniform() * *get_LENGTH();
			agents[i]->y = randomUniform() * *get_CIRCUMFERENCE();
			agents[i]->colour = *get_LTI_AGENT_TYPE();
            agents[i]->velocity = randomGaussian();
            //agents[i]->respond_x = NULL;
            //agents[i]->respond_y = NULL;
		}
        h_add_agents_LTi_lti_random_movement(agents, count);
        h_free_agent_LTi_array(&agents, count);
	}

	//Create LTins:
	// Can create upto h_agent_AoS_MAX agents in a single pass (the number allocated for) but the full amount does not have to be created.
	const float ltin_migration_rate = spaces * *get_PERCENT_LTIN_FROM_FACS() / steps_24h;
    unsigned int ltin_to_add = 0;
    if(ltin_migration_rate > 1){
        ltin_to_add = floor(ltin_migration_rate);
    }else if((step % 100) > (ltin_migration_rate * 100)){
        //^ modulo with 100 for Percentage:
        ltin_to_add = 1;
    }
	// It is sensible to check if it is possible to create new agents, and if so how many.
	agent_remaining = get_agent_LTin_MAX_count() - get_agent_LTin_ltin_random_movement_count();
	if (agent_remaining > 0) {
		unsigned int count = (ltin_to_add > agent_remaining) ? agent_remaining: ltin_to_add;
        xmachine_memory_LTin** agents = h_allocate_agent_LTin_array(count);
		// Populate data as required
		for (unsigned int i = 0; i < count; i++) {
			//agents[i] = h_allocate_agent_LTin();
			//Initialise agent variables:
			agents[i]->x = randomUniform() * *get_LENGTH();
			agents[i]->y = randomUniform() * *get_CIRCUMFERENCE();
			agents[i]->colour = *get_LTIN_AGENT_TYPE();
            agents[i]->velocity = randomGaussian();
		}
        h_add_agents_LTin_ltin_random_movement(agents, count);
        h_free_agent_LTin_array(&agents, count);
	}
}

/*
 * Movement function, calculates the new position for an agent travelling at the given velocity in a random direction.
 */
__FLAME_GPU_FUNC__ glm::vec2 random_move(glm::vec2 position,
                                         float velocity,
                                         RNG_rand48* rand48)
{
	//Calculate velocity
	float angle = 2 * M_PI * rnd<CONTINUOUS>(rand48);

	float x_move = MAX_CELL_SPEED * velocity * sinf(angle);
	float y_move = MAX_CELL_SPEED * velocity * cosf(angle);

	glm::vec2 agent_velocity = glm::vec2(x_move, y_move);

	position += agent_velocity;

	//agent should die if it goes outside area of the x direction
	if(position.x < 0 || LENGTH < position.x){
		position.x = NULL;
	}

	//Y position should wrap
	position.y = fmod(position.y, float(CIRCUMFERENCE));
	
	return position;
}

/**
 * random_move FLAMEGPU Agent Function
 * This method is responsible for the agent moving randomly around the plane.
 *
 * @param agent Pointer to an LTi agent. This represents a single agent instance and can be modified directly.
 * @param lto_location_messages Pointer to output message list of type xmachine_message_example_message_list.
 * 
 * @return Return type is always int. 0 means the agent does NOT die.
 */
__FLAME_GPU_FUNC__ int lti_random_move(xmachine_memory_LTi* agent,
    xmachine_message_chemokine_list* chemokine_messages,
    RNG_rand48* rand48)
{
    //Detect chemotine expression before moving!
    //If chemotine, then set respond_x, respond_y return.
    xmachine_message_chemokine* message = get_first_chemokine_message(chemokine_messages);

    const glm::vec2 init_position = glm::vec2(agent->x, agent->y);
    float threshold = 0;
    float respond_x;
    float respond_y;

    while(message){
        //Check if expression is sufficient enough..
        const glm::vec2 lto_position = glm::vec2(message->x, message->y);
        const float linear_adjust = message->linear_adjust;

        const float distance_to_lto = fabs(glm::distance(lto_position, init_position)); 
        const float level = chemokineLevel(distance_to_lto, linear_adjust);

        if(threshold < level){
            threshold = level;
            respond_x = message->x;
            respond_y = message->y;
        }

        message = get_next_chemokine_message(message, chemokine_messages);
    }

    //if(distance_to_lto < 50){
    if(CHEMO_THRESHOLD < threshold){
        //Calculate 3x3 grid for movement direction from Kieran's thesis..
        //Or even something more accurate?
        if(rnd<CONTINUOUS>(rand48) < threshold){
            agent->respond_x = respond_x;
            agent->respond_y = respond_y;
        }
    }
	
    glm::vec2 new_position = random_move(init_position, agent->velocity, rand48);

    //Agent DIES
    if(new_position.x == NULL){
    	return -1;
    }
    
    agent->x = new_position.x;
    agent->y = new_position.y;
    
    return 0;
}

/**
 * The agent is now bound.
 * It transitions to the chemotaxis state (see model file)
 */
__FLAME_GPU_FUNC__ int direction(xmachine_memory_LTi* agent){
    //printf("Chemotaxis\n");
    return 0;
}

/**
 * The cell is in the chemotaxis state
 * It moves directly towards its "respond_x" and "respond_y" position
 */
__FLAME_GPU_FUNC__ int direct_move(xmachine_memory_LTi* agent, RNG_rand48* rand48){
    //TODO- this should follow grid method, described in Kieran's paper
    const float x_aim = agent->respond_x;
    const float y_aim = agent->respond_y;

    const glm::vec2 target = glm::vec2(x_aim, y_aim);
    const glm::vec2 current = glm::vec2(agent->x, agent->y);
    const float dist_between = glm::distance(target, current);
   
    if(dist_between < ADHESION_DISTANCE_THRESHOLD){
        agent->stable_contact = 1;
        //printf("LTi Contact\n");
        return 0;
    }

    //Vector between
    glm::vec2 move = target - current;
    //Distance = 1
    move = glm::normalize(move);
    //Can travel up to this
    move *= agent->velocity * MAX_CELL_SPEED;

    agent->x += move.x;
    agent->y += move.y;
    //Y position should wrap
    agent->y = fmod(agent->y, float(CIRCUMFERENCE));
    
    return 0;
}

/**
 * The cell is in the adhesion state- no movement.
 */
__FLAME_GPU_FUNC__ int contact(xmachine_memory_LTi* agent){
    //Random check to determine if the cell should escape adhesion

    return 0;
}

/**
 * The agent has escaped adhesion
 * this transition changes state (see model file) to random_move
 */
__FLAME_GPU_FUNC__ int check_escape(xmachine_memory_LTi* agent, RNG_rand48* rand48){
    if(rnd<CONTINUOUS>(rand48) < 0.5){
        agent->respond_x = NULL;
        agent->respond_y = NULL;
    }

    return 0;
}

/**
 * Cell transitions from adhesion to random movement
 * See Model File
 */
__FLAME_GPU_FUNC__ int lti_escape(xmachine_memory_LTi* agent){
    return 0;
}

/**
 * The agent is in the random movement state.
 * It should move in a random direction at the predefined speed.
 */
__FLAME_GPU_FUNC__ int ltin_random_move(xmachine_memory_LTin* xmemory,
                                        xmachine_message_lto_location_list* lto_location_messages,
                                        xmachine_message_lto_location_PBM* partition_matrix, 
                                        RNG_rand48* rand48)
{
    xmachine_message_lto_location* message = get_first_lto_location_message(lto_location_messages, partition_matrix,
    	xmemory->x, xmemory->y, 0.0);
    while(message){
        //Check if BIND is SUFFICIENT - 50% is THRESHOLD BIND PROBABILITY
   		if(rnd<CONTINUOUS>(rand48) < 0.5){
   			//Transition to STABLE CONTACT
   			xmemory->stable_contact = 1;
            //TODO: Here we should ALSO message LTo
            //to begin upregulation of adhesion molecules
   		}


    	message = get_next_lto_location_message(message, lto_location_messages, partition_matrix);
    }

	glm::vec2 init_position = glm::vec2(xmemory->x, xmemory->y);
    glm::vec2 new_position = random_move(init_position, xmemory->velocity, rand48);
    //Agent DIES
    if(new_position.x == NULL){
    	return -1;
    }
    
    xmemory->x = new_position.x;
    xmemory->y = new_position.y;
    
    return 0;
}

__FLAME_GPU_FUNC__ int ltin_adhesion(xmachine_memory_LTin* agent,
    xmachine_message_ltin_location_list* ltin_location_messages,
    RNG_rand48* rand48)
{
    //Placeholder transition for movement between states
    //This transition logic is performed in the XMLModelFile.xml

    //Alert LTo cell of collision:
    add_ltin_location_message(ltin_location_messages,
        agent->x,
        agent->y,
        0.0
    );

	return 0;
}

__FLAME_GPU_FUNC__ int ltin_escape(xmachine_memory_LTin* agent){
    return 0;
}

__FLAME_GPU_FUNC__ int ltin_localised_move(xmachine_memory_LTin* agent, RNG_rand48* rand48){
    //Move locally around the LTo cell.

    //If escape becomes more likely
    if(rnd<CONTINUOUS>(rand48) < 0.5){
        agent->stable_contact = 0;
    }

	return 0;
}

__FLAME_GPU_FUNC__ int output_location(xmachine_memory_LTo* agent,
							   xmachine_message_lto_location_list* lto_location_messages)
{
	add_lto_location_message(lto_location_messages,
        agent->x,
        agent->y,
        0.0,
        agent->adhesion_probability
    );
    
    return 0;
}

__FLAME_GPU_FUNC__ int output_location2(xmachine_memory_LTo* agent,
    xmachine_message_lto_location_list* lto_location_messages){
    //Duplicate of above function for CHEMOTINE state
    add_lto_location_message(lto_location_messages,
        agent->x,
        agent->y,
        0.0,
        agent->adhesion_probability
    );
    
    return 0;
}

/**
 *  Perform generic cell division:
 */
__FLAME_GPU_FUNC__ int divide(xmachine_memory_LTo* agent,
    xmachine_memory_LTo_list* LTo_agents,
    RNG_rand48* rand48){

    float x = agent->x - LTO_CELL_SIZE * 2;
    float y = agent->y + rnd<CONTINUOUS>(rand48);

    add_LTo_agent(LTo_agents,
        x, y, LTO_AGENT_TYPE,
        CHEMO_UPPER_ADJUST, agent->adhesion_probability, 60 * 12
    );

    return 0;
}

__FLAME_GPU_FUNC__ int begin_chemokine(xmachine_memory_LTo* agent){
    //Placeholder transition:
    //LTo moves to emit chemokine state
    return 0;
}

__FLAME_GPU_FUNC__ int express_chemokine(xmachine_memory_LTo* agent,
    xmachine_message_ltin_location_list* ltin_location_messages,
    xmachine_message_ltin_location_PBM* partition_matrix,
    xmachine_message_chemokine_list* chemokine_messages)
{
    //Update linear curve value:
    xmachine_message_ltin_location* message = get_first_ltin_location_message(
        ltin_location_messages, partition_matrix,
        agent->x, agent->y, 0
    );
    while(message){
        agent->linear_adjust -= INCREASE_CHEMO_EXPRESSION;
        message = get_next_ltin_location_message(message, ltin_location_messages, partition_matrix);
    }

    add_chemokine_message(chemokine_messages,
        agent->x,
        agent->y,
        agent->linear_adjust
    );

    return 0;
}

__FLAME_GPU_FUNC__ int detect_collision(xmachine_memory_LTo* agent,
    xmachine_message_ltin_location_list* ltin_location_messages,
    xmachine_message_ltin_location_PBM* partition_matrix)
{
    //Check for location message:
    //If true, begin chemokine emission.
    xmachine_message_ltin_location* message = get_first_ltin_location_message(
        ltin_location_messages, partition_matrix,
        agent->x, agent->y, 0
    );

    //Iterate through these messages:
    while(message){
        //calculate how to resolve the positions!
        agent->linear_adjust = CHEMO_UPPER_ADJUST;
        message = get_next_ltin_location_message(message, ltin_location_messages, partition_matrix);
    }

    return 0;
}

__FLAME_GPU_FUNC__ int mature(xmachine_memory_LTo* agent){
    return 0;
}

/**
 *  Output forces, to help resolve overlapping states
 */
__FLAME_GPU_FUNC__ int output_force(xmachine_memory_LTo* agent, xmachine_message_force_list* force_messages){
    add_force_message(force_messages,
        agent->x,
        agent->y,
        0.0
    );

    return 0;
}

/**
 *  Resolve overlapping states
 */
__FLAME_GPU_FUNC__ int resolve(xmachine_memory_LTo* agent,
    xmachine_message_force_list* force_messages,
    xmachine_message_force_PBM* partition_matrix){

    int cells_above = 0;
    int cells_below = 0;

    xmachine_message_force* message = get_first_force_message(
        force_messages, partition_matrix,
        agent->x, agent->y, 0
    );

    //Iterate through these messages:
    while(message){
        //calculate how to resolve the positions!
        if(agent->y < message->y){
            cells_above++;
        }else{
            cells_below++;
        }

        message = get_next_force_message(message, force_messages, partition_matrix);
    }

    float y_diff = (cells_below * LTO_CELL_SIZE) - (cells_above * LTO_CELL_SIZE);
    agent->y += y_diff;
    agent->y = fmod(agent->y, float(CIRCUMFERENCE));

    return 0;
}

__FLAME_GPU_FUNC__ int resolved(xmachine_memory_LTo* agent){
    return 0;
}

#endif //_FLAMEGPU_FUNCTIONS
