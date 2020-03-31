
class SkeletonVis{
  Joint [] joints;
  Table table;
  
  SkeletonVis(String file){
      joints = new Joint[22];
      int[] jointSizes = {30,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10, 1, 1};
      String[] names = {"head", "neck", "spine", "hip", "shoulderL", "shoulderR", "elbowL", "elbowR", "wristL", "wristR", "handL", "handR", "handtipL", "handtipR", "hipL", "hipR", "kneeL", "kneeR", "ankleL", "ankleR", "footL", "footR"};
      for (int i = 0; i < names.length; i++){
        joints[i] = new Joint(names[i], jointSizes[i]);
      }
      table = loadTable(file, "header");
  }
  //lock the positions of hipR and hipL to hip 
  void changeJointLocations(){
    TableRow row = table.getRow(frameCount%table.getRowCount());
    for (int i = 0; i < joints.length; i++){
      float x = row.getFloat(joints[i].name + "_x");
      float y = row.getFloat(joints[i].name + "_y");
      float z = row.getFloat(joints[i].name + "_z");
      
      joints[i].changeLocation(x, y, z);
    }
  }
  
  void drawJoints(){
    for (int i = 0; i < joints.length; i++){
      if (joints[i].name != "handtipL"){
        if (joints[i].name != "handtipR"){
          joints[i].display();
        }
      }
    }
  }
  
  void drawBone(Joint a, Joint b){
    //size(100, 100, P3D);
    stroke(126);
    //translate(300, - 600, - 5000);
    line(a.x + 300, a.y - 300, a.z - 5000, b.x + 300, b.y - 300, b.z - 5000);
  }
  
  void drawBody(){
    drawJoints();
    //head and torso
    drawBone(joints[0], joints[1]); //head to neck
    drawBone(joints[1], joints[2]); //neck to spine
    drawBone(joints[1], joints[4]); //neck to shoulderL
    drawBone(joints[1], joints[5]);//neck to shoulderR
    drawBone(joints[2], joints[3]); //spine to hip
    drawBone(joints[3], joints[14]); //hip to hipL
    drawBone(joints[3], joints[15]); //hip to hipR
    
    //left arm
    drawBone(joints[4], joints[6]); //shoulderL to elbowL
    drawBone(joints[6], joints[8]); //elbowL to wristL 
    drawBone(joints[8], joints[10]); //wristL to handL
    //drawBone(joints[10], joints[12]); //handL to handtipL
    
    //right arm
    drawBone(joints[5], joints[7]); //shoulderR to elbowR
    drawBone(joints[7], joints[9]); //elbowR to wristR
    drawBone(joints[9], joints[11]); //wristR to handR
    //drawBone(joints[11], joints[13]); //handR to handtipR
    
    //left leg
    drawBone(joints[14], joints[16]); //hipL to kneeL
    drawBone(joints[16], joints[18]); //kneeL to ankleL
    drawBone(joints[18], joints[20]); //ankleL to footL
    //right leg
    drawBone(joints[15], joints[17]); //hipR to kneeR
    drawBone(joints[17], joints[19]); //kneeR to ankleR
    drawBone(joints[19], joints[21]); //ankleR to footR
  }
}
