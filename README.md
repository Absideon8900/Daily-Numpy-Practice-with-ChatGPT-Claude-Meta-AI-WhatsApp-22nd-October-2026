# Daily Numpy Practice with ChatGPT, Claude, Meta AI (WhatsApp) 22nd October 2026
<b>NumPy GIS Challenge — Task 1: Vegetation Detection</b></br>
You're working as a GIS Analyst studying vegetation distribution in a particular area using satellite-derived NDVI data.</br>
You receive this raster:</br>
```Python
import numpy as np
ndvi = np.array([
    [0.12, 0.65, 0.72, 0.30],
    [0.81, 0.45, 0.22, 0.56],
    [0.33, 0.91, 0.15, 0.68],
    [0.74, 0.28, 0.52, 0.88]
])
```
The raster contains <b>16 pixels arranged in 4 rows and 4 columns.</b> Each pixel stores an NDVI value representing vegetation-related information derived from satellite imagery.</br>

For this exercise, your classification rule is:</br>
<ul>
  <li>NDVI values <b>greater than</b> or <b>equal to 0.50</b> are classified as vegetation.</li>
  <li>NDVI values below 0.50 are classified as non-vegetation.</li>
</ul>

This threshold is an assumption for our exercise, not a universal rule for vegetation classification.</br>

<b>🎯 Your task 1:</b></br>
Create a Boolean array named vegetation_mask that identifies which pixels satisfy the vegetation classification rule.</br>

Your output must show:</br>
<ul>
  <li><code>True</code> for pixels classified as vegetation.</li>
  <li><code>False</code> for pixels classified as non-vegetation.</li>
</ul>

<b>🎯 Your task 2:</b></br>
Calculate vegetation area</br>
Now imagine you're preparing a GIS report.</br>
Your study area has 16 pixels, and each pixel represents <b>10 m</b> × <b>10 m.</b></br>
Calculate the total area classified as vegetation in square metres.</br>
Requirements:</br>
<ul>
  <li>Use your existing <code>vegetation_mask.</code></li>
  <li>Use NumPy to calculate the area.</li>
  <li>Store the final answer in a variable named <code>vegetation_area.</code></li>
  <li>Print the result.</li>
</ul>

<b>🎯 Your task 3:</b></br>
Vegetation percentage</br>
Your GIS report now needs one more result: <b>What percentage of the study area's total classified area is vegetation?</b></br>
You already have all the variables you need.</br>
Write the code yourself and store the result in a variable called <code>vegetation_percentage.</code></br>

<b>🎯 Your task 4:</b></br>
A more realistic GIS problem</br>
Now imagine your faculty asks:</br>
<b>“What is the average NDVI of vegetation pixels only?”</b></br>
This is different from calculating the average NDVI of the entire study area. We only want pixels satisfying your existing vegetation threshold, NDVI ≥ 0.50.</br>
Your next task is to calculate that average and store it in <code>mean_vegetation_ndvi.</code></br>
Use your existing <code>ndvi</code> array and the NumPy concepts you've already practised.</br>

<b>🎯 Your task 5:</b></br>
Find the highest NDVI vegetation pixel</br>
<b>“Where is the pixel with the highest NDVI value among the pixels classified as vegetation?”</b></br>
Find both:</br>
<ul>
  <li>The highest NDVI value among pixels where NDVI ≥ 0.50.</li>
  <li>Its row and column indices in the original array.</li>
</ul>
Use the NumPy concepts you've already learned. Keep the original NDVI values intact.</br>

<b>🎯 Your task 6:</b></br>
Now, let's finish the original challenge</br>
Your supervisor asked for the row and column indices of the highest NDVI vegetation pixel.</br>
You have already found its value: <code>0.91.</code></br>
Here is where your earlier NumPy practice becomes useful. Remember <code>np.argmax()</code> and <code>np.unravel_index()?</code></br>
The challenge is to apply them to the vegetation pixels while retaining their original positions in the 4 × 4 raster.</br>

Why does this matter in GIS? A maximum value alone tells you what the highest value is. Its row and column tell you where in the raster it occurs. To obtain real-world coordinates, you'd then use the raster's georeferencing information.</br>

Your next step: Add code to find the row and column of the highest vegetation pixel. Don't rewrite the whole script—just add the few lines needed for this task. 😈🛰️</br>

# Solution (Script)
```Python
import numpy as np
ndvi = np.array([
    [0.12, 0.65, 0.72, 0.30],
    [0.81, 0.45, 0.22, 0.56],
    [0.33, 0.91, 0.15, 0.68],
    [0.74, 0.28, 0.52, 0.88]
])
vegetation_mask = np.where(ndvi>=0.50,1,0)
the_count_of_pixels = np.size(vegetation_mask)
the_vegetation_total_no_of_pixels = np.sum(vegetation_mask) 
the_non_vegetation_total_no_of_pixels = the_count_of_pixels - the_vegetation_total_no_of_pixels
the_vegetation_total_area = the_vegetation_total_no_of_pixels*100 
the_non_vegetation_total_area = the_non_vegetation_total_no_of_pixels*100 
the_percentage_of_vegetation_area_coverage = (the_vegetation_total_area / (the_vegetation_total_area + the_non_vegetation_total_area))* 100
the_percentage_of_non_vegetation_area_coverage = (the_non_vegetation_total_area / (the_vegetation_total_area + the_non_vegetation_total_area)) * 100
vegetation = ndvi.copy()
non_vegetation = ndvi.copy()
average_vegetation = np.mean(vegetation[vegetation>=0.5])
average_non_vegetation = np.mean(non_vegetation[non_vegetation<0.5])
max_vegetation = np.max(vegetation[vegetation>=0.5])
min_vegetation = np.min(vegetation[vegetation>=0.5])
max_non_vegetation = np.max(non_vegetation[non_vegetation<0.5])
min_non_vegetation = np.min(non_vegetation[non_vegetation<0.5])

max_vegatation_index = np.argmax(vegetation[vegetation>=0.5])
max_vegatation_position = np.unravel_index(max_vegatation_index,np.shape(ndvi))
min_vegetation_index = np.argmin(vegetation[vegetation>=0.5])
min_vegetation_position  = np.unravel_index(min_vegetation_index,np.shape(ndvi))

max_non_vegatation_index = np.argmax(non_vegetation[non_vegetation<0.5])
max_non_vegatation_position = np.unravel_index(max_non_vegatation_index,np.shape(ndvi))
min_non_vegetation_index = np.argmin(non_vegetation[non_vegetation<0.5])
min_non_vegetation_position = np.unravel_index(min_non_vegetation_index,np.shape(ndvi))

print(vegetation_mask)
print()
print("This would determine the total no. pixels:")
print(the_count_of_pixels) 
print()
print()
print("FOR VEGETATION")
print("The total no. of vegetation based pixels:")
print(the_vegetation_total_no_of_pixels)
print("The total area of vegetation based pixels:")
print(the_vegetation_total_area,"sq/m")
print("The percentage of area of vegetation covers the rest of the area:")
print(the_percentage_of_vegetation_area_coverage,"%")
print("The average pixel which are more & equals to 0.5")
print(average_vegetation)
print("The highest level of pixel which is more & equals to 0.5")
print(max_vegetation)
print("The lowest level of pixel which is more & equals to 0.5")
print(min_vegetation)
print("The highest level of pixel which is more & equals to 0.5 position in the 2D array with respect to row & column:")
print(max_vegatation_position)
print("The lowest level of pixel which is more & equals to 0.5 position in the 2D array with respect to row & column")
print(min_vegetation_position)
print()
print()
print("FOR NON-VEGETATION")
print("The total no. of non-vegetation based pixels:")
print(the_non_vegetation_total_no_of_pixels)
print("The total area of non-vegetation based pixels:")
print(the_non_vegetation_total_area,"sq/m")
print("The percentage of area of non-vegetation covers the rest of the area:")
print(the_percentage_of_non_vegetation_area_coverage,"%")
print("The average pixel which are less than 0.5")
print(average_non_vegetation)
print("The highest level of pixel which are less than 0.5")
print(max_non_vegetation)
print("The lowest level of pixel which are less than 0.5")
print(min_non_vegetation)
print("The highest level of pixel which are less than 0.5 position in the 2D array with respect to row & column:")
print(max_non_vegatation_position)
print("The lowest level of pixel which are less than 0.5 position in the 2D array with respect to row & column:")
print(min_non_vegetation_position)
```

# Solution (Output)
```
PS C:\Users\hp\Downloads\sql_practice(0)> & C:\Users\hp\AppData\Local\Programs\Python\Python313\python.exe "c:/Users/hp/Downloads/sql_practice(0)/test2.py"
[[0 1 1 0]
 [1 0 0 1]
 [0 1 0 1]
 [1 0 1 1]]

This would determine the total no. pixels:
16


FOR VEGETATION
The total no. of vegetation based pixels:
9
The total area of vegetation based pixels:
900 sq/m
The percentage of area of vegetation covers the rest of the area:
56.25 %
The average pixel which are more & equals to 0.5
0.7188888888888889
The highest level of pixel which is more & equals to 0.5
0.91
The lowest level of pixel which is more & equals to 0.5
0.52
The highest level of pixel which is more & equals to 0.5 position in the 2D array with respect to row & column:
(np.int64(1), np.int64(0))
The lowest level of pixel which is more & equals to 0.5 position in the 2D array with respect to row & column
(np.int64(1), np.int64(3))


FOR NON-VEGETATION
The total no. of non-vegetation based pixels:
7
The total area of non-vegetation based pixels:
700 sq/m
The percentage of area of non-vegetation covers the rest of the area:
43.75 %
The average pixel which are less than 0.5
0.2642857142857143
The highest level of pixel which are less than 0.5
0.45
The lowest level of pixel which are less than 0.5
0.12
The highest level of pixel which are less than 0.5 position in the 2D array with respect to row & column:
(np.int64(0), np.int64(2))
The lowest level of pixel which are less than 0.5 position in the 2D array with respect to row & column:
(np.int64(0), np.int64(0))
PS C:\Users\hp\Downloads\sql_practice(0)> 
```
