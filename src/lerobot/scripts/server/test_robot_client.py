import grpc
import pickle
import time
from lerobot.transport import services_pb2, services_pb2_grpc, async_inference_pb2, async_inference_pb2_grpc
from lerobot.scripts.server.helpers import RemotePolicyConfig

def main():
    # 服务器地址（根据实际情况修改）
    SERVER_ADDRESS = "192.168.2.70:8080"
    
    # 创建gRPC通道和客户端存根
    channel = grpc.insecure_channel(SERVER_ADDRESS)
    # stub = services_pb2_grpc.PolicyServerStub(channel)
    stub = async_inference_pb2_grpc.AsyncInferenceStub(channel)
    
    try:
        print(f"尝试连接到策略服务器: {SERVER_ADDRESS}")
        
        # 1. 测试基础连接
        start_time = time.time()
        grpc.channel_ready_future(channel).result(timeout=5)
        print(f"✅ 连接成功 (耗时: {time.time()-start_time:.2f}秒)")
        
        # 2. 发送Ready信号（握手）
        print("发送握手信号...")
        # stub.Ready(services_pb2.Empty())
        stub.Ready(async_inference_pb2.Empty())
        print("✅ 握手成功")
        
        # 3. 发送策略配置
        print("发送策略配置...")
        policy_config = RemotePolicyConfig(
            policy_type="act",  # 使用简单测试策略
            pretrained_name_or_path="/home/smai/workspace/dc_dir/lerobot_0901_pybullet/outputs/train/act_0923_2/checkpoints/last/pretrained_model",
            device="cuda",
            actions_per_chunk=5,
            lerobot_features={"joints": ["j1", "j2", "j3"]}
        )
        # config_request = services_pb2.PolicyInstructions(
        #     data=pickle.dumps(policy_config)
        # )
        config_request = async_inference_pb2.PolicySetup(
            data=pickle.dumps(policy_config)
        )
        stub.SendPolicyInstructions(config_request)
        print("✅ 策略配置已发送")
        
        # 4. 发送模拟观测数据并获取动作
        print("\n开始测试数据交互...")
        for i in range(3):  # 测试3轮
            # 生成模拟观测
            mock_observation = {
                "timestamp": time.time(),
                "observation": {
                    "joint_positions": [0.1*i, 0.2*i, 0.3*i],
                    "joint_velocities": [0.01, 0.02, 0.03]
                },
                "robot_id": "test_robot"
            }
            
            # 发送观测
            # obs_request = services_pb2.Observations(
            #     data=pickle.dumps([mock_observation])
            # )
            obs_request = async_inference_pb2.Observation(
                data=pickle.dumps([mock_observation])
            )
            stub.SendObservations(obs_request)
            print(f"发送第{i+1}轮观测数据")
            
            # 获取动作
            # action_response = stub.GetActions(services_pb2.Empty())
            action_response = stub.GetActions(async_inference_pb2.Empty())
            actions = pickle.loads(action_response.data)
            print(f"收到动作: {actions[:2]}... (共{len(actions)}个动作)")
            
            time.sleep(1)  # 间隔1秒
        
        print("\n🎉 所有测试完成，通信正常！")
        
    except grpc.FutureTimeoutError:
        print("❌ 连接超时 - 服务器未响应，请检查服务器是否启动")
    except grpc.RpcError as e:
        print(f"❌ gRPC通信错误: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
    finally:
        channel.close()
        print("连接已关闭")

if __name__ == "__main__":
    main()
