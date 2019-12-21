package com.example.signreco

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.os.Bundle
import com.google.android.material.bottomnavigation.BottomNavigationView
import androidx.appcompat.app.AppCompatActivity
import androidx.navigation.findNavController
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.setupActionBarWithNavController
import androidx.navigation.ui.setupWithNavController

import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2
import org.opencv.android.CameraBridgeViewBase
import androidx.core.content.ContextCompat.getSystemService
import android.icu.lang.UCharacter.GraphemeClusterBreak.T
import android.view.SurfaceView
import androidx.core.content.ContextCompat.getSystemService
import android.icu.lang.UCharacter.GraphemeClusterBreak.T
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame
import org.opencv.core.Mat
import androidx.core.content.ContextCompat.getSystemService
import android.icu.lang.UCharacter.GraphemeClusterBreak.T
import androidx.core.content.ContextCompat.getSystemService
import android.icu.lang.UCharacter.GraphemeClusterBreak.T
import android.view.Menu
import android.view.MenuItem
import org.opencv.android.OpenCVLoader
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.BaseLoaderCallback
import androidx.core.content.ContextCompat.getSystemService
import android.icu.lang.UCharacter.GraphemeClusterBreak.T
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.nio.file.Files.size
import org.opencv.core.Core
import androidx.core.app.ComponentActivity
import androidx.core.app.ComponentActivity.ExtraData
import androidx.core.content.ContextCompat.getSystemService
import android.icu.lang.UCharacter.GraphemeClusterBreak.T
import android.R.attr.name
import android.R.attr.noHistory
import android.util.Log
import org.opencv.core.CvType
import org.opencv.imgproc.Imgproc
import org.opencv.video.BackgroundSubtractor
import org.opencv.video.BackgroundSubtractorMOG2
import org.opencv.video.Video


class MainActivity : AppCompatActivity(), CvCameraViewListener2 {
    private var mOpenCvCameraView: CameraBridgeViewBase? = null
    private var backgroundSub: BackgroundSubtractor? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        getCameraPermission()

        mOpenCvCameraView = findViewById(R.id.CameraView) as? CameraBridgeViewBase
        mOpenCvCameraView?.visibility = SurfaceView.VISIBLE
        mOpenCvCameraView?.setCvCameraViewListener(this)

        val navView: BottomNavigationView = findViewById(R.id.nav_view)
        val navController = findNavController(R.id.nav_host_fragment)
        // Passing each menu ID as a set of Ids because each
        // menu should be considered as top level destinations.
        val appBarConfiguration = AppBarConfiguration(setOf(R.id.navigation_home, R.id.navigation_dashboard, R.id.navigation_notifications))
        setupActionBarWithNavController(navController, appBarConfiguration)
        navView.setupWithNavController(navController)


        OpenCVLoader.initDebug()
        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)

        backgroundSub = Video.createBackgroundSubtractorKNN(500, 400.0, false)
    }

    private fun getCameraPermission() {
        // First check android version
        // if (MyVersion > Build.VERSION_CODES.LOLLIPOP_MR1) {
        // Check if permission is already granted

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                listOf(Manifest.permission.CAMERA).toTypedArray(),
                1)
        }
    }

    public override fun onPause() {
        super.onPause()
        mOpenCvCameraView?.disableView()
    }

    public override fun onDestroy() {
        super.onDestroy()
        mOpenCvCameraView?.disableView()
    }

    override fun onCameraViewStarted(width: Int, height: Int) {}

    override fun onCameraViewStopped() {}

    override fun onCameraFrame(inputFrame: CvCameraViewFrame): Mat {
        var frame = inputFrame.rgba()
        var mask = Mat()
        var kernel = Mat(4, 4, CvType.CV_8U)

        backgroundSub?.apply(frame, mask, 0.0)

        //Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel)
        //Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel)
        //Core.bitwise_and(frame, mask, frame)

        return mask
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        // Inflate the menu; this adds items to the action bar if it is present.
        // menuInflater.inflate(R.menu.main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        //val id = item.itemId
        //return if (id == R.id.action_settings) true else super.onOptionsItemSelected(item)
        return true
    }

    private val mLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                LoaderCallbackInterface.SUCCESS -> {
                    mOpenCvCameraView?.enableView()
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    public override fun onResume() {
        super.onResume()
        //OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback)
    }
}
