# Epsilon Tools

Epsilon has been used to create a GUI tool that can automatically generate FLAME simulation models.

The tools provide a set of well-formedness constraints (EVL) to allow non-technical users to validate the model within the graphical editor.

![GMF Validation](https://github.com/Oliver-Binns/PRIY/raw/master/Report/Appendix/validation_gmf.png)


These well-formedness constraints also implement quick-fixes to allow the user to quickly fix any validation issues with their model.

![GMF Quick Fix](https://github.com/Oliver-Binns/PRIY/raw/master/Report/Appendix/validation_quickfix_gmf.png)


The tools provide a model-to-text transformation (EGL) to allow the FLAME .xml model input files to be produced from the graphical editor.

In future, there is scope to provide migration tools (ETL/Flock) to migrate models to future versions of FLAME if required.

In order to create a simulation, a set of C source code files will also be needed to provide agent behaviour.

![PPSim Model in GMF](https://github.com/Oliver-Binns/PRIY/raw/master/Report/Appendix/ppsim_gmf.png)
