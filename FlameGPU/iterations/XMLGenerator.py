# Python is temporarily being used to generate sample XML.
# This code can be migrated to C/Cuda for speed-ups later in the project.

from lxml import etree
import random

LTin = 5
LTi = 5
LTo = 5

#Create XML
root = etree.Element('states')

itno = etree.Element('itno')
itno.text = "0"
root.append(itno)

#Setup Environment
environment = etree.Element('environment')

variables = {}
for k, v in variables.items():
	variable = etree.Element(k)
	variable.text = v
	environment.append(variable)

#root.append(environment)

def instantiateCell(variables):
	agent = etree.Element('xagent')

	for k, v in variables.items():
		variable = etree.Element(k)
		if not isinstance(v, str):
			v = str(v)
		variable.text = v
		agent.append(variable)

	return agent

pos = 0

# Unsure whether these agents should be identified by type, or have a global id
# currently they are identified by type
for i in range(LTin):
	root.append(instantiateCell({
		'name': 'LTin',
		'id': i,
		'x': pos,
		'y': 0,
		'velocity': round(random.random(), 5)
	}))
	pos += 1

for i in range(LTi):
	root.append(instantiateCell({
		'name': 'LTi',
		'id': i,
		'x': pos,
		'y': 0,
		'velocity': round(random.random(), 5)
	}))
	pos += 1

for i in range(LTo):
	root.append(instantiateCell({
		'name': 'LTo',
		'id': i,
		'x': pos,
		'y': 0
	}))
	pos += 1

tree = etree.ElementTree(root)
tree.write('0.xml')
