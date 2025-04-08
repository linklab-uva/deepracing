from deepracing_models.data_loading.dbf_ablations import DBFExperiment, DBFExperimentCollection
import matplotlib.pyplot as plt
import matplotlib.axes, matplotlib.figure, matplotlib.patches
import numpy as np
import torch
import os


def summary_plot(experiment : DBFExperiment, axes : matplotlib.axes.Axes):
    """
    Create a summary plot for a DBFExperiment.

    Args:
        experiment (DBFExperiment): The DBFExperiment object to plot.
        axes (matplotlib.axes.Axes): The axes to plot on.
    """
    innerbound_helper = experiment.innerbound_helper()
    outerbound_helper = experiment.outerbound_helper()
    
    datadict = experiment.datadict()
    
    particle_history=datadict["particle_history"]
    particle0 = particle_history[0]
    iteration_particles = particle_history[1:]
    
    
    
    