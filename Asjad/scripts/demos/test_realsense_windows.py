import pyrealsense2 as rs
import time

def main():
    print("---------------------------------------")
    print("RealSense Windows Diagnostics")
    print("---------------------------------------")
    
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("ERROR: No RealSense devices found!")
        print("Check your USB connection.")
        return

    for dev in devices:
        print(f"Device Found: {dev.get_info(rs.camera_info.name)}")
        print(f"  Serial: {dev.get_info(rs.camera_info.serial_number)}")
        print(f"  Firmware: {dev.get_info(rs.camera_info.firmware_version)}")
        
        # Check USB Type
        try:
           usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
           print(f"  USB Connection: {usb_type}")
           if "2." in usb_type:
               print("  [WARNING] Connected via USB 2.0! Streaming capabilities will be limited.")
               print("            Use a high-quality USB 3.0 (Type-C to Type-A/C) cable.")
        except:
           print("  USB Connection: Unknown")
           
    print("\nAttempting to start stream...")
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Try a modest resolution that works on USB 2.0 and 3.0
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        profile = pipeline.start(config)
        print("SUCCESS: Stream started!")
        
        # Check active profile
        active_depth = profile.get_stream(rs.stream.depth)
        print(f"Active Stream: {active_depth.as_video_stream_profile().width()}x{active_depth.as_video_stream_profile().height()} @ {active_depth.as_video_stream_profile().fps()}fps")
        
        print("Streaming 50 frames...")
        for i in range(50):
            frames = pipeline.wait_for_frames()
            if i % 10 == 0:
                print(f"  Frame {i}/50 received...")
                
        pipeline.stop()
        print("SUCCESS: Test Complete. Hardware is working.")
        print("---------------------------------------")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to stream: {e}")
        print("Possible causes:")
        print("1. USB Bandwidth (Try a different port/cable)")
        print("2. Camera is in a bad state (Unplug/Replug)")

if __name__ == "__main__":
    main()
