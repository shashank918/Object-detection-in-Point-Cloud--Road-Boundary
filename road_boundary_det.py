import pcl
import numpy as np
from datetime import datetime
import math
from geo2cart import geo2cart
from collections import defaultdict


class CleanData:
	def __init__(self):
		pass
	
	@staticmethod
	def intensity_points(): 
		cartesian_coord = []
		date = str(datetime.now()).replace(".","_").replace(":","_").strip(" ")
		print("Started Detection of Lane boundaries")
		with open('cartesian-{}.obj'.format(date),'w') as cartesian_file:
			with open('Filter.fuse', 'r') as fuse_file:
				for line in fuse_file:
					line = line.split()					
					cartesian_cord = geo2cart.cartesian(float(line[0]),float(line[1]),float(line[2]))
					cartesian_file.write("v {} {} {}\n".format(cartesian_cord[0],cartesian_cord[1],cartesian_cord[2]))
					cartesian_coord.append([cartesian_cord[0],cartesian_cord[1],cartesian_cord[2]])
		
		return cartesian_coord
						
class OutlierFilter:
	def __init__(self, mean, std_dev):
		self.mean = mean
		self.std_dev = std_dev
	
	@staticmethod
	def convert_list_to_numpy(combined_points):
		return np.array(combined_points, dtype=np.float32)
	
	def generate_cloud(self,pcl_points):
		pcl_object = pcl.PointCloud()
		pcl_object.from_array(pcl_points)
		print("Started Filtering Outlier points")
		outlier_filter = pcl_object.make_statistical_outlier_filter() # Filter class uses point neighborhood statistics to filter outlier data.
		outlier_filter.set_mean_k(self.mean)
		outlier_filter.set_std_dev_mul_thresh(self.std_dev)
		outlier_filter.set_negative(False)
		return outlier_filter.filter()
	
	@staticmethod
	def generate_outlier_file(outliers):
		print("Writing filtered points to a file")
		date = str(datetime.now()).replace(".","_").replace(":","_").strip(" ")
		with open("filter-{}.obj".format(date), 'w') as filter_file:
			for point in outliers:
				filter_file.write("v {} {} {}\n".format(point[0], point[1], point[2]))
	
	@staticmethod
	def clear_clustered_objects(outliers,index,thresh):
		get_nearest_neighbor_points = outliers.make_kdtree_flann()# KdTreeFLANN is a generic type of 3D spatial locator using kD-tree structures, the class is making use of the FLANN (Fast Library for Approximate Nearest Neighbor)
		k_indices, k_sqr_distances = get_nearest_neighbor_points.nearest_k_search_for_cloud(outliers, index)#Find the k nearest neighbours and squared distances for all points in the pointcloud.Results are in ndarrays, size (pc.size, k) Returns: (k_indices, k_sqr_distances)
		#print("KINDICES", k_indices)
		#print("KSQR",k_sqr_distances)
		distances = np.sum(k_sqr_distances, axis=1)
		#print("DIST", distances)
		
		blocks = []
		for i in range(np.shape(distances)[0]):
			if distances[i] < float(thresh):
				blocks.extend(k_indices[i]) 
		unique_indices = list(set(blocks))
		print("Clearing closely connected Objects such as Cars, trees etc.")
		print("Writing to Object-removed.obj")
		outliers = outliers.extract(unique_indices, negative=False)
		with open('Object-removed.obj', 'w') as object_cleared_file:
			for point in outliers:
				line = "v {} {} {}\n".format(point[0], point[1], point[2]) 
				object_cleared_file.write(line)
		return outliers	
	
	@staticmethod
	def cylobjrem(outliers,model,iter):
		cylobjrem_set = outliers.make_segmenter_normals(ksearch=50)# Return a pcl.SegmentationNormal object with this object set as the input-cloud
		cylobjrem_set.set_optimize_coefficients(True)
		cylobjrem_set.set_normal_distance_weight(0.1)
		cylobjrem_set.set_method_type(pcl.SAC_RANSAC)
		cylobjrem_set.set_max_iterations(iter)

		if model == pcl.SACMODEL_CYLINDER:
			cylobjrem_set.set_model_type(model)
			cylobjrem_set.set_distance_threshold(20)
			cylobjrem_set.set_radius_limits(0, 10)
			segmented_indices, model = cylobjrem_set.segment()
			Cylindrical_seg = outliers.extract(segmented_indices, negative=False)
			print("Generated Cylindrical segments")
			final = open('Cylinder.obj', 'w')
			for point in Cylindrical_seg:
				line = "v {} {} {}\n".format(str(point[0]), str(point[1]), str(point[2]))
				final.write(line)
			final.close()

			return Cylindrical_seg
		
	
	@staticmethod
	def final_filtered_points():
		longitude = []
		print("Removing poles")
		
		print("Writing data to final_data.obj")
		on_longitude = defaultdict(list)

		with open("Cylinder.obj", "r") as cylinder_file:
			for line in cylinder_file.readlines():
				line = line.split(" ")
				
				longitude.append(float(line[2]))
				on_longitude[float(line[2])].append([float(line[1]), float(line[2]), float(line[3].strip("\n"))])
			
		sorted_longitude = list(set(sorted(longitude)))[700:]

		with open("final_data.obj", "w") as final_file:
			for point in sorted_longitude:
				for value in on_longitude[point]:
					final_file.write("v {} {} {}\n".format(value[0],value[1],value[2]))

	
			
if __name__ == '__main__':
	index = 400
	thresh = 7000
	# file = open('final_project_point_cloud.fuse', 'r')
	# filter_file = open('Filter.fuse', 'w')
	# for line in file:
		# k = line.split()
		# if(int(k[3])>15):
			# filter_file.write(line)
	cartesian_coord = CleanData().intensity_points()
	
	coordinate_numpy = OutlierFilter.convert_list_to_numpy(cartesian_coord)
	
	outliers = OutlierFilter(50, 5)
	filtered_points = outliers.generate_cloud(coordinate_numpy)

	
	#OutlierFilter.generate_outlier_file(filtered_points)
	outliers = OutlierFilter.clear_clustered_objects(filtered_points,index,thresh)
	
	
	Cylindrical_seg = OutlierFilter.cylobjrem(outliers,model = pcl.SACMODEL_CYLINDER,iter = 500)
	OutlierFilter.final_filtered_points()
	