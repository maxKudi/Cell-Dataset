<h2 align="center">Cell-Dataset</h2>

[![GitHub stars](https://img.shields.io/github/stars/maxKudi/Cell-Dataset)](https://github.com/maxKudi/Cell-Dataset/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/maxKudi/Cell-Dataset)](https://github.com/maxKudi/Cell-Dataset/network)
[![GitHub issues](https://img.shields.io/github/issues/maxKudi/Cell-Dataset)](https://github.com/maxKudi/Cell-Dataset/issues)
[![GitHub license](https://img.shields.io/github/license/maxKudi/Cell-Dataset)](https://github.com/maxKudi/Cell-Dataset/blob/master/LICENSE)

The [```Living and Dead cell (LDC)```](https://github.com/maxKudi/Cell-Dataset/) dataset contains 2500 images along with their annotation files, the original dataset is not   split into ```Training```, ```Testing```, and ```Validation``` sets. I would recommend using [```roboflow```](roboflow.com) to augment the dataset. Among the 360 smear images, 300 blood cell images with annotations are used as the training set first, and then the rest of the 60 images with annotations are used as the testing set. Due to the shortage of the data, a subset of the training set is used to prepare the validation set which contains 60 images with annotations.

[![Download](https://img.shields.io/badge/download-dataset-f20a0a.svg?longCache=true&style=flat)](https://github.com/maxKudi/Cell-Dataset/archive/master.zip)

## Performance 
The Dataset still have some noticeable error so i would recommend picking the best images among the 2500 before perforing the augmentation 

## Data Description

### Image 
Each image is resized to ```225 x 219``` resolution. 
<p align="center">
  <img src="https://user-images.githubusercontent.com/22647359/127229072-5d6a41c8-7f9f-4e35-abc9-0460a37c9b64.png" width="500">
</p>

`N.B.` Rectangular bounding boxes are converted to circular bounding boxes for representation.

### Annotation Format

```
<annotation>
	<folder>Label</folder>
	<filename>b (568).png</filename>
	<path>C:\Users\Kudi Okerulu\Desktop\Label\b (568).png</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>225</width>
		<height>219</height>
		<depth>1</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>Living Cell</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>157</xmin>
			<ymin>83</ymin>
			<xmax>178</xmax>
			<ymax>105</ymax>
		</bndbox>
	</object>
</annotation>

```

[1]: http://ietdl.org/t/kmgztb

