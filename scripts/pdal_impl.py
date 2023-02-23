
import os
import sys

import subprocess

def chamfer_dist(pcdA, pcdB):
    """
    Computes the Chamfer Distance between two point clouds
    """
    # Compute the directed Chamfer distance from pcdA to pcdB
    output = subprocess.check_output(['pdal', 'chamfer', pcdA, pcdB]).decode('utf-8')
    output_dict = eval(output)
    chamfer_dist = output_dict['chamfer']
    #print("Chamfer Distance: \n\n", chamfer_dist)
    
    return chamfer_dist


def hausdorff_dist(pcdA, pcdB):
    """
    Computes the Hausdorff Distance between two point clouds
    """
    # Compute the directed Chamfer distance from pcdA to pcdB
    output = subprocess.check_output(['pdal', 'hausdorff', pcdA, pcdB]).decode('utf-8')
    output_dict = eval(output)
    #print(output_dict)
    hausdorff_dist = output_dict['hausdorff']
    #print("Hausdorff Distance: \n\n", hausdorff_dist)
    
    return hausdorff_dist



def main():

    # Load the two point clouds
    pcdA = sys.argv[1]
    pcdB = sys.argv[2]
    
    # Compute the Chamfer Distance
    hausdorff_dist(pcdA, pcdB)

if __name__ == "__main__":
    main()