
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
    const int lti_cell_size = *get_LTI_CELL_SIZE();
    const int lto_cell_size = *get_LTO_CELL_SIZE();

	int ADHESION_DISTANCE_THRESHOLD = (lti_cell_size + lto_cell_size) / 2;
	set_ADHESION_DISTANCE_THRESHOLD(&ADHESION_DISTANCE_THRESHOLD);	
}

__device__ float adhesionProbability(float lto_adhesion_level){
    float thresholded = fminf(MAX_ADHESION_PROBABILITY, lto_adhesion_level);
    return ADHESION_SLOPE * lto_adhesion_level;
}

__device__ float chemokineLevel(float distanceToLTo, float lto_expression_level){
    float thresholded = fmaxf(CHEMO_LOWER_ADJUST, lto_expression_level);
    float value = -(-thresholded * distanceToLTo + CHEMO_CURVE_ADJUST);
    value = expf(value);
    value = 1 + value;
    return 1 / value;
}

__device__ float distanceBetween(glm::vec2 a, glm::vec2 b){
    return fabs(glm::distance(a, b));
}

/*
 * Movement function, calculates the new position for an agent travelling at the given velocity in a random direction.
 */
__device__ glm::vec2 random_move(glm::vec2 position,
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
    if(position.y < 0){
        position.y += CIRCUMFERENCE;
    }else if(position.y > CIRCUMFERENCE){
        position.y -= CIRCUMFERENCE;
    }
    
    return position;
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

    if(step > 4321){
        exit(0);
    }

    //TODO: refactor LTi, LTin migration to share code, if possible:
	//CREATE LTis:
	// Can create upto h_agent_AoS_MAX agents in a single pass (the number allocated for) but the full amount does not have to be created.
    const float spaces = (*get_LENGTH() * *get_CIRCUMFERENCE()) / powf(*get_LTI_CELL_SIZE(), 2);
    const float steps_24h = 24.0f * 60.0f;

	const float lti_migration_rate = spaces * *get_PERCENT_LTI_FROM_FACS() / steps_24h;

    const float lti_entered = lti_migration_rate * (float)step;
    const float lti_entered_floor = floorf(lti_entered);
    if((lti_entered - lti_entered_floor) < lti_migration_rate){
        // It is sensible to check if it is possible to create new agents, and if so how many.
        unsigned int agent_remaining = get_agent_LTi_MAX_count() - get_agent_LTi_lti_random_movement_count();
        if (agent_remaining > 0) {
            unsigned int count = 1;
            xmachine_memory_LTi** agents = h_allocate_agent_LTi_array(count);
            // Populate data as required
            for (unsigned int i = 0; i < count; i++) {
                //agents[i] = h_allocate_agent_LTi();
                //Initialise agent variables:
                agents[i]->x = randomUniform() * *get_LENGTH();
                agents[i]->y = randomUniform() * *get_CIRCUMFERENCE();
                agents[i]->colour = *get_LTI_AGENT_TYPE();
                agents[i]->velocity = randomGaussian();
            }
            h_add_agents_LTi_lti_random_movement(agents, count);
            h_free_agent_LTi_array(&agents, count);
        }
    }
    
    
	//Create LTins:
	// Can create upto h_agent_AoS_MAX agents in a single pass (the number allocated for) but the full amount does not have to be created.
	const float ltin_migration_rate = spaces * *get_PERCENT_LTIN_FROM_FACS() / steps_24h;

    const float ltin_entered = ltin_migration_rate * (float)step;
    const float ltin_entered_floor = floorf(ltin_entered);
    if((ltin_entered - ltin_entered_floor) < ltin_migration_rate){
        // It is sensible to check if it is possible to create new agents, and if so how many.
        unsigned int agent_remaining = get_agent_LTin_MAX_count() - get_agent_LTin_ltin_random_movement_count();
        if (agent_remaining > 0) {
            unsigned int count = 1;
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
    //Detect chemokine expression before moving!
    //If chemokine, then set respond_x, respond_y return.
    xmachine_message_chemokine* message = get_first_chemokine_message(chemokine_messages);

    const glm::vec2 init_position = glm::vec2(agent->x, agent->y);
    float threshold = 0;
    float respond_x;
    float respond_y;

    while(message){
        //Check if expression is sufficient enough..
        const glm::vec2 lto_position = glm::vec2(message->x, message->y);
        const float linear_adjust = message->linear_adjust;


        const float distance_to_lto = distanceBetween(lto_position, init_position); 
        const float level = chemokineLevel(distance_to_lto, linear_adjust);

        if(threshold < level){
            threshold = level;
            respond_x = message->x;
            respond_y = message->y;
        }

        message = get_next_chemokine_message(message, chemokine_messages);
    }

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
   
    if(dist_between <= ADHESION_DISTANCE_THRESHOLD){
        agent->stable_contact = 1;
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
    agent->y = fmodf(agent->y, float(CIRCUMFERENCE));
    
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
__FLAME_GPU_FUNC__ int check_escape(xmachine_memory_LTi* agent,
    xmachine_message_lto_location_list* lto_location_messages,
    RNG_rand48* rand48){
    //Move locally around the LTo cell.
    xmachine_message_lto_location* message = get_first_lto_location_message(lto_location_messages);

    glm::vec2 agent_location = glm::vec2(agent->x, agent->y);

    while(message){
        glm::vec2 message_location = glm::vec2(agent->x, agent->y);

        //In contact with LTo
        if(distanceBetween(agent_location, message_location) <= ADHESION_DISTANCE_THRESHOLD){
           //If escape becomes more likely
            if(rnd<CONTINUOUS>(rand48) < adhesionProbability(message->adhesion_probability)){
                agent->respond_x = NULL;
                agent->respond_y = NULL;
            } 
        }
        
        message = get_next_lto_location_message(message, lto_location_messages);
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
                                        RNG_rand48* rand48)
{
    xmachine_message_lto_location* message = get_first_lto_location_message(lto_location_messages);

    glm::vec2 agent_location = glm::vec2(xmemory->x, xmemory->y);

    while(message){
        glm::vec2 message_location = glm::vec2(message->x, message->y);

        //Check if BIND is SUFFICIENT - 50% is THRESHOLD BIND PROBABILITY
        if(distanceBetween(agent_location, message_location) <= ADHESION_DISTANCE_THRESHOLD){

       		if(rnd<CONTINUOUS>(rand48) < 0.5){
       			//Transition to STABLE CONTACT
       			xmemory->stable_contact = 1;
       		}
        }

    	message = get_next_lto_location_message(message, lto_location_messages);
    }

    if(xmemory->stable_contact){
        return 0;
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
        agent->y
    );

	return 0;
}

__FLAME_GPU_FUNC__ int ltin_escape(xmachine_memory_LTin* agent){
    return 0;
}

__FLAME_GPU_FUNC__ int ltin_localised_move(xmachine_memory_LTin* agent,
    xmachine_message_lto_location_list* lto_location_messages,
    RNG_rand48* rand48){
    //Move locally around the LTo cell.
    xmachine_message_lto_location* message = get_first_lto_location_message(lto_location_messages);

    glm::vec2 agent_location = glm::vec2(agent->x, agent->y);

    while(message){
        glm::vec2 message_location = glm::vec2(message->x, message->y);

        if(distanceBetween(agent_location, message_location) <= ADHESION_DISTANCE_THRESHOLD){
        //If escape becomes more likely
            if(rnd<CONTINUOUS>(rand48) < adhesionProbability(message->adhesion_probability)){
                agent->stable_contact = 0;
            }
        }
        message = get_next_lto_location_message(message, lto_location_messages);
    }

	return 0;
}

__FLAME_GPU_FUNC__ int output_location(xmachine_memory_LTo* agent,
							   xmachine_message_lto_location_list* lto_location_messages)
{
	add_lto_location_message(lto_location_messages,
        agent->x,
        agent->y,
        agent->adhesion_probability
    );
    
    return 0;
}

__FLAME_GPU_FUNC__ int output_location2(xmachine_memory_LTo* agent,
    xmachine_message_lto_location_list* lto_location_messages){
    //Duplicate of above function for CHEMOKINE state
    add_lto_location_message(lto_location_messages,
        agent->x,
        agent->y,
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
        CHEMO_UPPER_ADJUST, INITIAL_ADHESION, (SIM_STEP - 1)
    );

    return 0;
}

__FLAME_GPU_FUNC__ int begin_chemokine(xmachine_memory_LTo* agent){
    //Placeholder transition:
    agent->created_at = SIM_STEP - 1;
    //LTo moves to emit chemokine state
    return 0;
}

__FLAME_GPU_FUNC__ int express_chemokine(xmachine_memory_LTo* agent,
    xmachine_message_ltin_location_list* ltin_location_messages,
    xmachine_message_chemokine_list* chemokine_messages)
{
    //Update linear curve value:
    xmachine_message_ltin_location* message = get_first_ltin_location_message(
        ltin_location_messages
    );

    glm::vec2 agent_location = glm::vec2(agent->x, agent->y);

    while(message){
        glm::vec2 message_location = glm::vec2(message->x, message->y);

        if(distanceBetween(agent_location, message_location) <= ADHESION_DISTANCE_THRESHOLD){
            agent->linear_adjust -= INCREASE_CHEMO_EXPRESSION;
            agent->adhesion_probability += ADHESION_INCREMENT;
        }

        message = get_next_ltin_location_message(message, ltin_location_messages);
    }

    add_chemokine_message(chemokine_messages,
        agent->x,
        agent->y,
        agent->linear_adjust
    );

    return 0;
}

__FLAME_GPU_FUNC__ int detect_collision(xmachine_memory_LTo* agent,
    xmachine_message_ltin_location_list* ltin_location_messages)
{
    //Check for location message:
    xmachine_message_ltin_location* message = get_first_ltin_location_message(
        ltin_location_messages
    );

    glm::vec2 agent_location = glm::vec2(agent->x, agent->y);
    //Iterate through these messages:
    while(message){
        glm::vec2 message_location = glm::vec2(message->x, message->y);
        //If true, begin chemokine emission.
        float dist = distanceBetween(agent_location, message_location);

        if(distanceBetween(agent_location, message_location) <= ADHESION_DISTANCE_THRESHOLD){
            agent->linear_adjust = CHEMO_UPPER_ADJUST;
        }

        message = get_next_ltin_location_message(message, ltin_location_messages);
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
        agent->y
    );

    return 0;
}

/**
 *  Resolve overlapping states
 */
__FLAME_GPU_FUNC__ int resolve(xmachine_memory_LTo* agent,
    xmachine_message_force_list* force_messages){

    int cells_above = 0;
    int cells_below = 0;

    xmachine_message_force* message = get_first_force_message(force_messages);

    glm::vec2 agent_location = glm::vec2(agent->x, agent->y);
    //Iterate through these messages:
    while(message){
        //calculate how to resolve the positions!
        glm::vec2 message_location = glm::vec2(message->x, message->y);

        if(distanceBetween(agent_location, message_location) <= ADHESION_DISTANCE_THRESHOLD){
            if(agent->y < message->y){
                cells_above++;
            }else{
                cells_below++;
            }
        }

        message = get_next_force_message(message, force_messages);
    }

    float y_diff = (cells_below * LTO_CELL_SIZE) - (cells_above * LTO_CELL_SIZE);
    agent->y += y_diff;
    agent->y = fmodf(agent->y, float(CIRCUMFERENCE));

    return 0;
}

__FLAME_GPU_FUNC__ int resolved(xmachine_memory_LTo* agent){
    return 0;
}

#endif //_FLAMEGPU_FUNCTIONS
