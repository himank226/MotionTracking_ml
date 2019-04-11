
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import org.opencv.core.*;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;


public class MotionTracking {

    private static BufferedImage Mat2BufferedImage(Mat img) {
        MatOfByte byteMat=new MatOfByte();
        Highgui.imencode(".jpg", img, byteMat);
        byte[] bytes=byteMat.toArray();
        InputStream in=new ByteArrayInputStream(bytes);
        BufferedImage image=null;
        try{
            image=ImageIO.read(in);
        }catch(IOException e){
            e.printStackTrace();
        }
        return image;
    }

    private static ArrayList<Rect> detect_contours(Mat diffFrame) {
        Mat v1=new Mat();
        Mat v2=diffFrame.clone();
        List<MatOfPoint> contours=new ArrayList<>();
        Imgproc.findContours(v2, contours, v1, Imgproc.RETR_LIST,
                Imgproc.CHAIN_APPROX_SIMPLE);
        double maxArea=100;
        int maxAreaIdx=-1;
        Rect r=null;
        ArrayList<Rect> rectArray=new ArrayList<>();
        for(int idx=0;idx<contours.size();idx++){
            Mat contour=contours.get(idx);
            double contourArea=Imgproc.contourArea(contour);
            if(contourArea>maxArea){
                maxAreaIdx=idx;
                r=Imgproc.boundingRect(contours.get(maxAreaIdx));
                rectArray.add(r);
                Imgproc.drawContours(img, contours, maxAreaIdx, new Scalar(0,0,255));
                
            }
        }
        v1.release();
        return rectArray;
    }
    static{
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
    static Mat img=null;
    public static void main(String args[]){
        JFrame vFrame=new JFrame("Motion Detector");
        vFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JLabel vLabel=new JLabel();
        vFrame.setContentPane(vLabel);
        vFrame.setSize(640, 480);
        vFrame.setVisible(true);
        
        Mat frame=new Mat();
        Mat outerBox= new Mat();
        Mat diffFrame=null;
        Mat tempFrame=null;
        
        ArrayList<Rect> array=new ArrayList<>();
        VideoCapture cam=new VideoCapture(0);
        Size sz=new Size(640,480);
        int i=0;
        
        while(true){
            if(cam.read(frame)){
                Imgproc.resize(frame,frame,sz);
                img=frame.clone();
                outerBox=new Mat(frame.size(),CvType.CV_8UC1);
                Imgproc.cvtColor(frame, outerBox, Imgproc.COLOR_BGR2GRAY);
                Imgproc.GaussianBlur(outerBox,outerBox, new Size(3,3), 0);
                if(i==0){
                    vFrame.setSize(frame.width(),frame.height());
                    diffFrame= new Mat(outerBox.size(),CvType.CV_8UC1);
                    tempFrame= new Mat(outerBox.size(),CvType.CV_8UC1);
                    diffFrame=outerBox.clone();
                }
                if(i==1){
                    Core.subtract(outerBox, tempFrame, diffFrame);
                    Imgproc.adaptiveThreshold(diffFrame, diffFrame, 255,
                            Imgproc.ADAPTIVE_THRESH_MEAN_C,
                            Imgproc.THRESH_BINARY_INV, 5, 2);
                    array=detect_contours(diffFrame);
                    if(array.size()>0){
                        Iterator<Rect> it=array.iterator();
                        while(it.hasNext()){
                            Rect obj=it.next();
                            Core.rectangle(img, obj.br(), obj.tl(),
                                    new Scalar(0,255,0),0);
                        }
                    }
                }
                i=1;
                ImageIcon image=new ImageIcon(Mat2BufferedImage(img));
                vLabel.setIcon(image);
                vLabel.repaint();
                tempFrame=outerBox.clone();
            }
        }
    }
}
