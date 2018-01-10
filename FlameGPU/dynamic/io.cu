
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

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <limits.h>
	

// include header
#include "header.h"

glm::vec3 agent_maximum;
glm::vec3 agent_minimum;

void readIntArrayInput(char* buffer, int *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = ",";
    char * token;

    token = strtok(buffer, s);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: Agent memory array has too many items, expected %d!\n", expected_items);
            exit(0);
        }
        
        array[i++] = atoi(token);
        
        token = strtok(NULL, s);
    }
    if (i != expected_items){
        printf("Error: Agent memory array has %d items, expected %d!\n", i, expected_items);
        exit(0);
    }
}

void readFloatArrayInput(char* buffer, float *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = ",";
    char * token;

    token = strtok(buffer, s);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: Agent memory array has too many items, expected %d!\n", expected_items);
            exit(0);
        }
        
        array[i++] = (float)atof(token);
        
        token = strtok(NULL, s);
    }
    if (i != expected_items){
        printf("Error: Agent memory array has %d items, expected %d!\n", i, expected_items);
        exit(0);
    }
}

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_LTin_list* h_LTins_ltin_random_movement, xmachine_memory_LTin_list* d_LTins_ltin_random_movement, int h_xmachine_memory_LTin_ltin_random_movement_count,xmachine_memory_LTin_list* h_LTins_stable_contact, xmachine_memory_LTin_list* d_LTins_stable_contact, int h_xmachine_memory_LTin_stable_contact_count,xmachine_memory_LTin_list* h_LTins_localised_movement, xmachine_memory_LTin_list* d_LTins_localised_movement, int h_xmachine_memory_LTin_localised_movement_count,xmachine_memory_LTi_list* h_LTis_lti_random_movement, xmachine_memory_LTi_list* d_LTis_lti_random_movement, int h_xmachine_memory_LTi_lti_random_movement_count,xmachine_memory_LTi_list* h_LTis_responding, xmachine_memory_LTi_list* d_LTis_responding, int h_xmachine_memory_LTi_responding_count,xmachine_memory_LTi_list* h_LTis_contact, xmachine_memory_LTi_list* d_LTis_contact, int h_xmachine_memory_LTi_contact_count,xmachine_memory_LTi_list* h_LTis_adhesion, xmachine_memory_LTi_list* d_LTis_adhesion, int h_xmachine_memory_LTi_adhesion_count,xmachine_memory_LTo_list* h_LTos_no_expression, xmachine_memory_LTo_list* d_LTos_no_expression, int h_xmachine_memory_LTo_no_expression_count,xmachine_memory_LTo_list* h_LTos_expression, xmachine_memory_LTo_list* d_LTos_expression, int h_xmachine_memory_LTo_expression_count,xmachine_memory_LTo_list* h_LTos_adhesion_upregulation, xmachine_memory_LTo_list* d_LTos_adhesion_upregulation, int h_xmachine_memory_LTo_adhesion_upregulation_count,xmachine_memory_LTo_list* h_LTos_chemokine_upregulation, xmachine_memory_LTo_list* d_LTos_chemokine_upregulation, int h_xmachine_memory_LTo_chemokine_upregulation_count,xmachine_memory_LTo_list* h_LTos_mature, xmachine_memory_LTo_list* d_LTos_mature, int h_xmachine_memory_LTo_mature_count,xmachine_memory_LTo_list* h_LTos_downregulated, xmachine_memory_LTo_list* d_LTos_downregulated, int h_xmachine_memory_LTo_downregulated_count)
{
	cudaError_t cudaStatus;
	
	//Device to host memory transfer
	
	cudaStatus = cudaMemcpy( h_LTins_ltin_random_movement, d_LTins_ltin_random_movement, sizeof(xmachine_memory_LTin_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying LTin Agent ltin_random_movement State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_LTins_stable_contact, d_LTins_stable_contact, sizeof(xmachine_memory_LTin_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying LTin Agent stable_contact State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_LTins_localised_movement, d_LTins_localised_movement, sizeof(xmachine_memory_LTin_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying LTin Agent localised_movement State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_LTis_lti_random_movement, d_LTis_lti_random_movement, sizeof(xmachine_memory_LTi_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying LTi Agent lti_random_movement State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_LTis_responding, d_LTis_responding, sizeof(xmachine_memory_LTi_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying LTi Agent responding State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_LTis_contact, d_LTis_contact, sizeof(xmachine_memory_LTi_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying LTi Agent contact State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_LTis_adhesion, d_LTis_adhesion, sizeof(xmachine_memory_LTi_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying LTi Agent adhesion State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_LTos_no_expression, d_LTos_no_expression, sizeof(xmachine_memory_LTo_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying LTo Agent no_expression State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_LTos_expression, d_LTos_expression, sizeof(xmachine_memory_LTo_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying LTo Agent expression State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_LTos_adhesion_upregulation, d_LTos_adhesion_upregulation, sizeof(xmachine_memory_LTo_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying LTo Agent adhesion_upregulation State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_LTos_chemokine_upregulation, d_LTos_chemokine_upregulation, sizeof(xmachine_memory_LTo_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying LTo Agent chemokine_upregulation State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_LTos_mature, d_LTos_mature, sizeof(xmachine_memory_LTo_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying LTo Agent mature State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_LTos_downregulated, d_LTos_downregulated, sizeof(xmachine_memory_LTo_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying LTo Agent downregulated State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	
	/* Pointer to file */
	FILE *file;
	char data[100];

	sprintf(data, "%s%i.xml", outputpath, iteration_number);
	//printf("Writing iteration %i data to %s\n", iteration_number, data);
	file = fopen(data, "w");
	fputs("<states>\n<itno>", file);
	sprintf(data, "%i", iteration_number);
	fputs(data, file);
	fputs("</itno>\n", file);
	fputs("<environment>\n" , file);
	fputs("</environment>\n" , file);

	//Write each LTin agent to xml
	for (int i=0; i<h_xmachine_memory_LTin_ltin_random_movement_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>LTin</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_LTins_ltin_random_movement->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_LTins_ltin_random_movement->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each LTin agent to xml
	for (int i=0; i<h_xmachine_memory_LTin_stable_contact_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>LTin</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_LTins_stable_contact->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_LTins_stable_contact->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each LTin agent to xml
	for (int i=0; i<h_xmachine_memory_LTin_localised_movement_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>LTin</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_LTins_localised_movement->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_LTins_localised_movement->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each LTi agent to xml
	for (int i=0; i<h_xmachine_memory_LTi_lti_random_movement_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>LTi</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_LTis_lti_random_movement->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_LTis_lti_random_movement->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each LTi agent to xml
	for (int i=0; i<h_xmachine_memory_LTi_responding_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>LTi</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_LTis_responding->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_LTis_responding->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each LTi agent to xml
	for (int i=0; i<h_xmachine_memory_LTi_contact_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>LTi</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_LTis_contact->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_LTis_contact->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each LTi agent to xml
	for (int i=0; i<h_xmachine_memory_LTi_adhesion_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>LTi</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_LTis_adhesion->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_LTis_adhesion->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each LTo agent to xml
	for (int i=0; i<h_xmachine_memory_LTo_no_expression_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>LTo</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_LTos_no_expression->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_LTos_no_expression->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each LTo agent to xml
	for (int i=0; i<h_xmachine_memory_LTo_expression_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>LTo</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_LTos_expression->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_LTos_expression->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each LTo agent to xml
	for (int i=0; i<h_xmachine_memory_LTo_adhesion_upregulation_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>LTo</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_LTos_adhesion_upregulation->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_LTos_adhesion_upregulation->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each LTo agent to xml
	for (int i=0; i<h_xmachine_memory_LTo_chemokine_upregulation_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>LTo</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_LTos_chemokine_upregulation->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_LTos_chemokine_upregulation->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each LTo agent to xml
	for (int i=0; i<h_xmachine_memory_LTo_mature_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>LTo</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_LTos_mature->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_LTos_mature->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each LTo agent to xml
	for (int i=0; i<h_xmachine_memory_LTo_downregulated_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>LTo</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_LTos_downregulated->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_LTos_downregulated->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("</xagent>\n", file);
	}
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);
}

void readInitialStates(char* inputpath, xmachine_memory_LTin_list* h_LTins, int* h_xmachine_memory_LTin_count,xmachine_memory_LTi_list* h_LTis, int* h_xmachine_memory_LTi_count,xmachine_memory_LTo_list* h_LTos, int* h_xmachine_memory_LTo_count)
{

	int temp = 0;
	int* itno = &temp;

	/* Pointer to file */
	FILE *file;
	/* Char and char buffer for reading file to */
	char c = ' ';
	char buffer[10000];
	char agentname[1000];

	/* Pointer to x-memory for initial state data */
	/*xmachine * current_xmachine;*/
	/* Variables for checking tags */
	int reading, i;
	int in_tag, in_itno, in_name;
    int in_LTin_x;
    int in_LTin_y;
    int in_LTi_x;
    int in_LTi_y;
    int in_LTo_x;
    int in_LTo_y;

	/* for continuous agents: set agent count to zero */	
	*h_xmachine_memory_LTin_count = 0;	
	*h_xmachine_memory_LTi_count = 0;	
	*h_xmachine_memory_LTo_count = 0;
	
	/* Variables for initial state data */
	float LTin_x;
	float LTin_y;
	float LTi_x;
	float LTi_y;
	float LTo_x;
	float LTo_y;
	
	/* Open config file to read-only */
	if((file = fopen(inputpath, "r"))==NULL)
	{
		printf("Error opening initial states\n");
		exit(0);
	}
	
	/* Initialise variables */
    agent_maximum.x = 0;
    agent_maximum.y = 0;
    agent_maximum.z = 0;
    agent_minimum.x = 0;
    agent_minimum.y = 0;
    agent_minimum.z = 0;
	reading = 1;
	in_tag = 0;
	in_itno = 0;
	in_name = 0;
	in_LTin_x = 0;
	in_LTin_y = 0;
	in_LTi_x = 0;
	in_LTi_y = 0;
	in_LTo_x = 0;
	in_LTo_y = 0;
	//set all LTin values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_LTin_MAX; k++)
	{	
		h_LTins->x[k] = 0;
		h_LTins->y[k] = 0;
	}
	
	//set all LTi values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_LTi_MAX; k++)
	{	
		h_LTis->x[k] = 0;
		h_LTis->y[k] = 0;
	}
	
	//set all LTo values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_LTo_MAX; k++)
	{	
		h_LTos->x[k] = 0;
		h_LTos->y[k] = 0;
	}
	

	/* Default variables for memory */
    LTin_x = 0;
    LTin_y = 0;
    LTi_x = 0;
    LTi_y = 0;
    LTo_x = 0;
    LTo_y = 0;

	/* Read file until end of xml */
    i = 0;
	while(reading==1)
	{
		/* Get the next char from the file */
		c = (char)fgetc(file);
		
		/* If the end of a tag */
		if(c == '>')
		{
			/* Place 0 at end of buffer to make chars a string */
			buffer[i] = 0;
			
			if(strcmp(buffer, "states") == 0) reading = 1;
			if(strcmp(buffer, "/states") == 0) reading = 0;
			if(strcmp(buffer, "itno") == 0) in_itno = 1;
			if(strcmp(buffer, "/itno") == 0) in_itno = 0;
			if(strcmp(buffer, "name") == 0) in_name = 1;
			if(strcmp(buffer, "/name") == 0) in_name = 0;
			if(strcmp(buffer, "/xagent") == 0)
			{
				if(strcmp(agentname, "LTin") == 0)
				{		
					if (*h_xmachine_memory_LTin_count > xmachine_memory_LTin_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent LTin exceeded whilst reading data\n", xmachine_memory_LTin_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(0);
					}
                    
					h_LTins->x[*h_xmachine_memory_LTin_count] = LTin_x;//Check maximum x value
                    if(agent_maximum.x < LTin_x)
                        agent_maximum.x = (float)LTin_x;
                    //Check minimum x value
                    if(agent_minimum.x > LTin_x)
                        agent_minimum.x = (float)LTin_x;
                    
					h_LTins->y[*h_xmachine_memory_LTin_count] = LTin_y;//Check maximum y value
                    if(agent_maximum.y < LTin_y)
                        agent_maximum.y = (float)LTin_y;
                    //Check minimum y value
                    if(agent_minimum.y > LTin_y)
                        agent_minimum.y = (float)LTin_y;
                    
					(*h_xmachine_memory_LTin_count) ++;	
				}
				else if(strcmp(agentname, "LTi") == 0)
				{		
					if (*h_xmachine_memory_LTi_count > xmachine_memory_LTi_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent LTi exceeded whilst reading data\n", xmachine_memory_LTi_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(0);
					}
                    
					h_LTis->x[*h_xmachine_memory_LTi_count] = LTi_x;//Check maximum x value
                    if(agent_maximum.x < LTi_x)
                        agent_maximum.x = (float)LTi_x;
                    //Check minimum x value
                    if(agent_minimum.x > LTi_x)
                        agent_minimum.x = (float)LTi_x;
                    
					h_LTis->y[*h_xmachine_memory_LTi_count] = LTi_y;//Check maximum y value
                    if(agent_maximum.y < LTi_y)
                        agent_maximum.y = (float)LTi_y;
                    //Check minimum y value
                    if(agent_minimum.y > LTi_y)
                        agent_minimum.y = (float)LTi_y;
                    
					(*h_xmachine_memory_LTi_count) ++;	
				}
				else if(strcmp(agentname, "LTo") == 0)
				{		
					if (*h_xmachine_memory_LTo_count > xmachine_memory_LTo_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent LTo exceeded whilst reading data\n", xmachine_memory_LTo_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(0);
					}
                    
					h_LTos->x[*h_xmachine_memory_LTo_count] = LTo_x;//Check maximum x value
                    if(agent_maximum.x < LTo_x)
                        agent_maximum.x = (float)LTo_x;
                    //Check minimum x value
                    if(agent_minimum.x > LTo_x)
                        agent_minimum.x = (float)LTo_x;
                    
					h_LTos->y[*h_xmachine_memory_LTo_count] = LTo_y;//Check maximum y value
                    if(agent_maximum.y < LTo_y)
                        agent_maximum.y = (float)LTo_y;
                    //Check minimum y value
                    if(agent_minimum.y > LTo_y)
                        agent_minimum.y = (float)LTo_y;
                    
					(*h_xmachine_memory_LTo_count) ++;	
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}
				

				
				/* Reset xagent variables */
                LTin_x = 0;
                LTin_y = 0;
                LTi_x = 0;
                LTi_y = 0;
                LTo_x = 0;
                LTo_y = 0;

			}
			if(strcmp(buffer, "x") == 0) in_LTin_x = 1;
			if(strcmp(buffer, "/x") == 0) in_LTin_x = 0;
			if(strcmp(buffer, "y") == 0) in_LTin_y = 1;
			if(strcmp(buffer, "/y") == 0) in_LTin_y = 0;
			if(strcmp(buffer, "x") == 0) in_LTi_x = 1;
			if(strcmp(buffer, "/x") == 0) in_LTi_x = 0;
			if(strcmp(buffer, "y") == 0) in_LTi_y = 1;
			if(strcmp(buffer, "/y") == 0) in_LTi_y = 0;
			if(strcmp(buffer, "x") == 0) in_LTo_x = 1;
			if(strcmp(buffer, "/x") == 0) in_LTo_x = 0;
			if(strcmp(buffer, "y") == 0) in_LTo_y = 1;
			if(strcmp(buffer, "/y") == 0) in_LTo_y = 0;
			
			
			/* End of tag and reset buffer */
			in_tag = 0;
			i = 0;
		}
		/* If start of tag */
		else if(c == '<')
		{
			/* Place /0 at end of buffer to end numbers */
			buffer[i] = 0;
			/* Flag in tag */
			in_tag = 1;
			
			if(in_itno) *itno = atoi(buffer);
			if(in_name) strcpy(agentname, buffer);
			else
			{
				if(in_LTin_x){ 
                    LTin_x = (float) atof(buffer);    
                }
				if(in_LTin_y){ 
                    LTin_y = (float) atof(buffer);    
                }
				if(in_LTi_x){ 
                    LTi_x = (float) atof(buffer);    
                }
				if(in_LTi_y){ 
                    LTi_y = (float) atof(buffer);    
                }
				if(in_LTo_x){ 
                    LTo_x = (float) atof(buffer);    
                }
				if(in_LTo_y){ 
                    LTo_y = (float) atof(buffer);    
                }
				
			}
			
			/* Reset buffer */
			i = 0;
		}
		/* If in tag put read char into buffer */
		else if(in_tag)
		{
			buffer[i] = c;
			i++;
		}
		/* If in data read char into buffer */
		else
		{
			buffer[i] = c;
			i++;
		}
	}
	/* Close the file */
	fclose(file);
}

glm::vec3 getMaximumBounds(){
    return agent_maximum;
}

glm::vec3 getMinimumBounds(){
    return agent_minimum;
}

