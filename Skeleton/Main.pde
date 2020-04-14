
import java.lang.Math;
import java.util.Arrays;
import java.time.LocalDateTime;


int frameR;
String fileName;
int canvasW = 500;
int canvasH = 500;
PGraphics canvas;

SkeletonVis dancer;

public void settings() {
  frameR = 15;
  size(1000, 1000, P3D);
  //pixelDensity(2);
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
