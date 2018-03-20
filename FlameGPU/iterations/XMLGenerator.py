# Python is temporarily being used to generate sample XML.
# This code can be migrated to C/Cuda for speed-ups later in the project.

from lxml import etree
import random

LTo_TYPE = 0;

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


for i in range(1):
	root.append(instantiateCell({
		'name': 'LTo',
		'id': i,
		'x': 7303 / 2,
		'y': 254 / 2,
		'colour': LTo_TYPE
	}))

tree = etree.ElementTree(root)
tree.write('0.xml')
