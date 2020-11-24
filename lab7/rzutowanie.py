import cv2, numpy as np
import os
import glob
import plyfile

camera_matrix =  np.array(
[[640.,   0., 320.],
 [  0., 640., 240.],
 [  0.,   0.,   1.]])

distortion_coeffs = np.zeros(4)


# Rzutowanie chmury punktów
data = plyfile.PlyData.read('./drill.ply')['vertex']
model_points = np.c_[data['x'], data['y'], data['z']]
model_points = model_points[::100]
model_points = model_points.astype(np.float32)

imgs = cv2.imread('00005.png')

rotation_vec = np.array([1.5719230120869594, -0.008669725486452024, 0.008394141727375536])
translation_vec = np.array([-0.20275877848305968, -0.09921681582613295, 5.319904180506665])

projected_points, _ = cv2.projectPoints(model_points, rotation_vec, translation_vec, camera_matrix, distortion_coeffs)

projected_points = projected_points.astype(np.int)
projected_points = projected_points.reshape((-1,2))

for point in projected_points:
    imgs[point[1],point[0]] = [255,255,255]

cv2.imwrite('result.png', imgs)




# Rzutowanie prostopadłościanu

cube_line_id = np.array([[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]]) #id from vertices

drill_cube_3d = np.array([[ 0.8291879 ,  0.22237803,  0.711129  ],
	   [ 0.8291879 ,  0.22237803, -0.693669  ],
	   [ 0.8291879 , -0.19283897,  0.711129  ],
	   [ 0.8291879 , -0.19283897, -0.693669  ],
	   [-0.52624696,  0.22237803,  0.711129  ],
	   [-0.52624696,  0.22237803, -0.693669  ],
	   [-0.52624696, -0.19283897,  0.711129  ],
	   [-0.52624696, -0.19283897, -0.693669  ]])  #max and min from model



def draw_cube(cube_3d, rot_vec, tran_vec, img_path, save_name):
    img = cv2.imread(img_path)

    cube_drill_2d, jacobian = cv2.projectPoints(cube_3d, rot_vec, tran_vec, camera_matrix, distortion_coeffs)
    cube_drill_2d = cube_drill_2d.astype(int)
    for line_id in cube_line_id:
        cv2.line(img,tuple(cube_drill_2d[line_id[0],0]),tuple(cube_drill_2d[line_id[1],0]), (255, 255, 255), 4)
    cv2.imwrite(save_name, img)


draw_cube(drill_cube_3d, rotation_vec, translation_vec, "00005.png", 'img_cube_0.png')



# Szkielet PSO
"""
W = 0.729
c1 = 2.05
c2 = 2.05

while iteration < n_iterations:
    for i in range(n_particles):
        fitness_cadidate = fitness_function()       

        if(pbest_fitness_value[i] > fitness_cadidate):
            pbest_fitness_value[i] = fitness_cadidate
            pbest_position[i] = particle_position_vector[i]

        if(gbest_fitness_value > fitness_cadidate):
            gbest_fitness_value = fitness_cadidate
            gbest_position = particle_position_vector[i]

    if(abs(gbest_fitness_value - target) < target_error):
        break

    for i in range(n_particles):
        new_velocity = (W*velocity_vector[i]) + (c1*random.random()) * (pbest_position[i] - particle_position_vector[i]) + (c2*random.random()) * (gbest_position-particle_position_vector[i])
        new_position = new_velocity + particle_position_vector[i]
        particle_position_vector[i] = new_position

    iteration = iteration + 1

"""