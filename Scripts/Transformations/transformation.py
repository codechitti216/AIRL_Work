import torch

Input_Coordinate_System = input()

print(Input_Coordinate_System)

Output_Coordinate_System = input().lower()

match Input_Coordinate_System:
    case "eci":
        match Output_Coordinate_System:
            case "ecef":
                print("success")