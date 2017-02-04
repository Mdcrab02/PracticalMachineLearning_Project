# PracticalMachineLearning_Project

## Introduction

This repo contains the course project for the Practical Machine Learning class from Johns Hopkins University.

## Purpose

The purpose of this project is to take accelerometer data recorded by test subjects performing various activities,
learn the values associated with a type of activity, and predict that type of activity.

## Method

Because the data contained feature vectors that were almost all NULL, they had to be removed before predictions
could be made.  From there, Principle Component Analysis (PCA) was used to reduce the high dimensionality of the
data but preserve the information contained within.

## The Data

The data for this project is the metrics from accelerometers placed on test subjects doing various activities.

The data can be found here: [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har)
