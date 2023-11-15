import numpy as np, copy, os, re, argparse
from calibration_library import *
from dataParsing_library import *
from debug_test import *
from distortion_library import * 
from icp_library import *

def main(): 
    # User interface prompt that takes input from user
    parser = argparse.ArgumentParser(description='homework_2 input')
    parser.add_argument('input_type', help='The debug or unknown input data to process')
    parser.add_argument('choose_set', help='The alphabetical index of the data set')
    args = parser.parse_args()

    # Read in input dataset
    script_directory = os.path.dirname(__file__)
    dirname = os.path.dirname(script_directory)
    base_path = os.path.join(dirname, f'PROGRAMS\\2023_pa345_student_data\\PA3-{args.input_type}-{args.choose_set}') 

    #Prolem3-BodyA.txt - 6 markers on Frame A and 1 tip 
    PA3_BodyA = 'PROGRAMS\\2023_pa345_student_data\\2023 PA345 Student Data\\Problem3-BodyA.txt'

    #Prolem3-BodyB.txt - 6 markers on Frame B and 1 tip
    PA3_BodyB = 'PROGRAMS\\2023_pa345_student_data\\2023 PA345 Student Data\\Problem3-BodyB.txt'

    #Problem3Mesh.sur - 1568 vertices and 3135 triangles (3 vertices index denoted as P Q R)
    PA3_Mesh = 'PROGRAMS\\2023_pa345_student_data\\2023 PA345 Student Data\\Problem3Mesh.sur'
    # parse two dataset 
    # for second data set, keep first 3 datapoints

    #SampleReadingsTest
    SampleReading = base_path + '-SampleReadingsTest.txt'
    SampleReading_point_cloud = parseData(SampleReading)
    SampleReading_frames = parseFrame(SampleReading_point_cloud, 6+6+4) # 15 frames of 16 points, ignore last 4 for PA3

    registration = setRegistration()
    np.set_printoptions(formatter={'float': '{:.2f}'.format})

    # Step 1
    # find tip position in relation to Frame B 
    
    find_tip_positions()
    # a_frames = 15 frames * 6 points - 1 set 
    # b_frames = same 
    # led_a = PA3_BodyA first 6 points 
    # led_b = same 
    # tip_a = PA3_Body last point

    # return 15 tip_position_b
    
    # Step 2



    output = []
    for i in range(4):
        trans_FG = registration.apply_transformation_single_pt(p_tip_G, trans_matrix_FG_nav[i])
        output.append(trans_FG)
    
    for i in range(4):
        output[i] = registration.apply_transformation_single_pt(output[i], np.linalg.inv(transformation_matrix_Bj))

    # format output
    output_name = f'pa2-{args.input_type}-{args.choose_set}-output2.txt'

    max_length = max(max(len(f"{point[0]:.2f}"), len(f"{point[1]:.2f}"), len(f"{point[2]:.2f}")) for point in output)

    with open(output_name, "w") as file:
        file.write('4, ' + output_name + '\n')

        for point in output:
            formatted_point = '  ' + f"{ point[0]:>{max_length}.2f}, {point[1]:>{max_length}.2f}, {point[2]:>{max_length}.2f}\n"
            file.write(formatted_point)     
 
if __name__ == "__main__":
    main()
    
    # v = validate()
    # file1 = 'pa1_student_data\PA1 Student Data\pa1-debug-g-output1.txt'
    # file2 = 'OUTPUT\pa1-unknown-g-output.txt'
    
    # percentage_differences = v.calculate_error_from_sample(file1, file2, use_reference=0)
    # print(np.mean(percentage_differences))
    
    