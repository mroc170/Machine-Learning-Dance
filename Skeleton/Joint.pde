class Joint{
  float x;
  float y;
  float z;
  int size;
  
  String name;
  //int num;
  
  Joint(String name, int size){
    this.name = name;
    this.size = size;
  }
  
  void display() {
    pushMatrix();
    //canvas.fill();
    lights();
    noStroke();
    translate((float)x + 300, (float)y - 300, (float)z - 5000);
    sphere(size);
    popMatrix();
  }
  
  void changeLocation(float x, float y, float z){
    this.x = x*800;
    this.y = y*-800;
    this.z = z*800;
  }
  
}
    
    
