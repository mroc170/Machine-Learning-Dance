
import java.lang.Math;
import java.util.Arrays;
import java.time.LocalDateTime;


import peasy.*;
PeasyCam cam;
int frameR;
String fileName;
int canvasW = 500;
int canvasH = 500;
PGraphics canvas;

SkeletonVis dancer;

public void settings() {
  frameR = 24;
  size(1500, 1500, P3D);
  pixelDensity(2);
  smooth(8);
  
  //canvas = createGraphics(canvasW, canvasH, P3D);
  fileName = "test.csv";
  //frameRate(frameR);
  //fileName = 
  dancer = new SkeletonVis(fileName);
  
}


void draw() {
  background(0);
  
  dancer.changeJointLocations();
  dancer.drawBody();

}
