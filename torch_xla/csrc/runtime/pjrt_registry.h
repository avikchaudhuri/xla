#ifndef XLA_CLIENT_INITIALIZE_PJRT_CLIENT_H_
#define XLA_CLIENT_INITIALIZE_PJRT_CLIENT_H_

#include "torch_xla/csrc/runtime/xla_coordinator.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"


namespace xla {
namespace {

// forward declear
class PjRtCompileOnlyDevice;

class PjRtCompileOnlyClient;

class PjRtCompileOnlyBuffer;

class PjRtCompileOnlyExternalReference : public PjRtBuffer::ExternalReference {
 public:
  PjRtCompileOnlyExternalReference(PjRtCompileOnlyClient* client, PjRtCompileOnlyBuffer* buffer,
                            void* data_ptr)
      : client_(client), buffer_(buffer) {
    data_ptr_ = data_ptr;
  }
  ~PjRtCompileOnlyExternalReference() override;

 private:
  PjRtCompileOnlyClient* client_;
  PjRtCompileOnlyBuffer* buffer_;
};

class PjRtCompileOnlyBuffer : public PjRtBuffer {
  public:
    PjRtCompileOnlyBuffer(const Shape& shape, PjRtCompileOnlyDevice* device, PjRtCompileOnlyClient* client) {
      shape_ = shape;
      device_ = device;
      client_ = client;
    }
    const Shape& on_device_shape() const override;
    PjRtMemorySpace* memory_space() const override;
    PjRtDevice* device() const override;
    PjRtClient* client() const override;
    StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference() override;
    PjRtFuture<Status> ToLiteral(MutableLiteralBase* literal) override;
    PjRtFuture<Status> LazyToLiteral(absl::AnyInvocable<absl::StatusOr<MutableLiteralBase*>() &&> generator) override;
    StatusOr<size_t> GetOnDeviceSizeInBytes() const override;
    PjRtFuture<Status> CopyRawToHost(void* dst, int64_t offset, int64_t transfer_size) override;
    void Delete() override;
    // class ExternalReference {
    // public:
    //   virtual ~ExternalReference() = 0;
    //   // Return opaque device memory pointer to root buffer.
    //   void* OpaqueDeviceMemoryDataPointer() const { return data_ptr_; }

    //   // Stream is platform-specific. This is intended to support dlpack on GPU
    //   // and is not expected to be implemented for all hardware platforms.
    //   virtual Status WaitUntilBufferReadyOnStream(std::intptr_t stream) {
    //     return Unimplemented(
    //         "WaitUntilBufferReadyOnStream is only implemented for GPU.");
    //   }

    // protected:
    //   void* data_ptr_;
    // };
    StatusOr<std::unique_ptr<ExternalReference>> ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) override;
    bool IsDeleted() override;
    StatusOr<std::unique_ptr<PjRtBuffer>> CopyToDevice(PjRtDevice* dst_device) override;
    StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(PjRtMemorySpace* dst_memory_space) override;
    void CopyToRemoteDevice(PjRtFuture<StatusOr<std::string>> serialized_descriptor, RemoteSendCallback on_done) override;
    void CopyToRemoteDeviceScattered(
      PjRtFuture<StatusOr<std::vector<std::string>>> serialized_descriptors,
      std::vector<RemoteSendCallback> callbacks,
      const ScatterDetails& scatter_details) override;
    PjRtFuture<Status> GetReadyFuture() override;
    bool IsOnCpu() const override;

  private:
    Shape shape_;
    PjRtCompileOnlyDevice* device_;
    PjRtCompileOnlyClient* client_;
};

class PjRtCompileOnlyDevice : public PjRtDevice {
 public:
  explicit PjRtCompileOnlyDevice(const PjRtDeviceDescription* description);

  const PjRtDeviceDescription& description() const override;

  PjRtClient* client() const override;
  void SetClient(PjRtClient* client);
  bool IsAddressable() const override;
  int local_hardware_id() const override;

  PjRtLocalDeviceId local_device_id() const override;

  PjRtLocalHardwareId local_hardware_id_typed() const override;
  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override;
  Status TransferToInfeed(const LiteralSlice& literal) override;
  Status TransferFromOutfeed(MutableBorrowingLiteral literal) override;
  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;
  StatusOr<PjRtMemorySpace*> default_memory_space() const override;


  const PjRtDeviceDescription* description_;
  PjRtClient* client_;
};

class CompileOnlyPjRtClient final : public xla::PjRtClient {
public:
  explicit CompileOnlyPjRtClient(std::shared_ptr<PjRtTopologyDescription> topology);
  // implement those pure virtual methods:
  int device_count() const override;
  int addressable_device_count() const override;
  absl::Span<PjRtDevice* const> devices() const override;
  // PIZ: if we don't implement this, this will raise error in GetDefaultDevice()
  absl::Span<PjRtDevice* const> addressable_devices() const override;
  int process_index() const override;

  StatusOr<PjRtDevice*> LookupDevice(
        PjRtGlobalDeviceId global_device_id) const override;
  
   StatusOr<PjRtDevice*> LookupAddressableDevice(
        PjRtLocalDeviceId local_device_id) const override; 
  
    absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

    PjRtRuntimeType runtime_type() const override;

    absl::string_view platform_name() const override;
    absl::string_view platform_version() const override;
    PjRtPlatformId platform_id() const override;

    StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;
    StatusOr<Layout> GetDefaultLayout(PrimitiveType element_type,
                                            absl::Span<const int64_t> dims) override;
                                        
    StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis() const override;

    StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override;

    StatusOr<std::unique_ptr<PjRtExecutable>> CompileUnloaded(
      const XlaComputation& computation, CompileOptions options);

    StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      mlir::ModuleOp module, CompileOptions options) override;

    StatusOr<std::unique_ptr<PjRtLoadedExecutable>> DeserializeExecutable(
        absl::string_view serialized, std::optional<CompileOptions> options);

    StatusOr<std::unique_ptr<PjRtBuffer>> CreateUninitializedBuffer(
        const Shape& shape, PjRtDevice* device) override;


    StatusOr<std::unique_ptr<AsyncHostToDeviceTransferManager>>
    CreateBuffersForAsyncHostToDevice(absl::Span<const Shape> shapes,
                                    PjRtMemorySpace* memory_space) override;


    StatusOr<std::unique_ptr<AsyncHostToDeviceTransferManager>>
      CreateBuffersForAsyncHostToDevice(absl::Span<const Shape> shapes,
                                    PjRtDevice* device) override;



    StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtDevice* device) override;




    StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtDevice* device) override;


    StatusOr<std::unique_ptr<PjRtBuffer>> CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtDevice* device,
      std::function<void()> on_delete_callback,
      std::optional<std::intptr_t> stream) override;

    StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
      MakeCrossHostReceiveBuffers(absl::Span<const Shape> shapes,
                              PjRtDevice* device,
                              PjRtCrossHostRecvNotifier notifier) override;

    StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
      MakeCrossHostReceiveBuffersForGather(
        absl::Span<const Shape> shapes, std::vector<GatherDetails> gather_details,
        PjRtDevice* device, PjRtCrossHostRecvNotifier notifier) override;
    
    StatusOr<ChannelHandle> CreateChannelHandle() override;

    StatusOr<ChannelHandle> CreateDeviceToHostChannelHandle() override;

    StatusOr<ChannelHandle> CreateHostToDeviceChannelHandle() override;
    
    Status Defragment() override;


  private:
    std::shared_ptr<PjRtTopologyDescription> topology_;
    //  InvalidIfrtCompiler default_compiler_;
    std::vector<std::unique_ptr<const PjRtDeviceDescription>> descriptions_;
    std::vector<std::unique_ptr<PjRtCompileOnlyDevice>> owned_devices_;
    std::vector<PjRtDevice*> devices_; 
};

}}
namespace torch_xla {
namespace runtime {

class PjRtPlugin {
 public:
  virtual std::string library_path() const = 0;

  virtual const std::unordered_map<std::string, xla::PjRtValueType>
  client_create_options() const = 0;

  virtual bool requires_xla_coordinator() const = 0;
};

void RegisterPjRtPlugin(std::string name,
                        std::shared_ptr<const PjRtPlugin> plugin);

std::tuple<std::unique_ptr<xla::PjRtClient>, std::unique_ptr<XlaCoordinator>>
InitializePjRt(const std::string& device_type);

}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_CLIENT_INITIALIZE_PJRT_H_
