import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
from pathlib import Path


def set_chinese_font():

    font_options = [
        'SimHei',
        'Microsoft YaHei',
        'DejaVu Sans',
        'sans-serif'
    ]

    for font_name in font_options:
        try:
            matplotlib.rcParams['font.sans-serif'] = [font_name]
            matplotlib.rcParams['axes.unicode_minus'] = False
            print(f"使用字体: {font_name}")
            break
        except:
            continue



set_chinese_font()


def calculate_total_loss(csv_path, loss_components=None):

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None, None


    if loss_components is None:

        default_losses = [
            'train/box_loss', 'box_loss', 'train_box_loss',
            'train/cls_loss', 'cls_loss', 'train_cls_loss',
            'train/dfl_loss', 'dfl_loss', 'train_dfl_loss',
            'train/obj_loss', 'obj_loss', 'train_obj_loss'
        ]


        found_components = []
        for col in default_losses:
            if col in df.columns:
                found_components.append(col)


        if len(found_components) < 2:

            for col in df.columns:
                col_lower = col.lower()
                if 'loss' in col_lower and 'val' not in col_lower and 'metrics' not in col_lower:
                    if col not in found_components:
                        found_components.append(col)

        loss_components = found_components

    print(f"检测到用于计算总损失的组件 ({len(loss_components)}个):")
    for i, col in enumerate(loss_components, 1):
        print(f"  {i}. {col}")


    total_loss = pd.Series(0, index=df.index, dtype=np.float64)
    for col in loss_components:
        if col in df.columns:
            data = pd.to_numeric(df[col], errors='coerce')
            data = data.fillna(0).astype(np.float64)
            total_loss += data
            print(f"  - {col}: 已添加到总损失，范围[{data.min():.6f}, {data.max():.6f}]")


    total_loss_df = pd.DataFrame({
        'epoch': df['epoch'] if 'epoch' in df.columns else np.arange(len(df)),
        'total_loss': total_loss,
        'loss_change': total_loss.diff().fillna(0),
        'loss_smooth': total_loss.rolling(window=5, min_periods=1).mean()
    })

    return total_loss_df, loss_components


def plot_total_loss_3d_simple(csv_path, save_path=None, start_epoch=0, end_epoch=None, figsize=(14, 10)):

    total_loss_df, loss_components = calculate_total_loss(csv_path)
    if total_loss_df is None:
        return None, None

    print(f"总损失计算完成，数据形状: {total_loss_df.shape}")
    print(f"总损失范围: [{total_loss_df['total_loss'].min():.4f}, {total_loss_df['total_loss'].max():.4f}]")


    if end_epoch is None:
        end_epoch = len(total_loss_df) - 1

    mask = (total_loss_df['epoch'] >= start_epoch) & (total_loss_df['epoch'] <= end_epoch)
    filtered_data = total_loss_df[mask].copy()

    if len(filtered_data) == 0:
        print("错误: 没有符合条件的数据")
        return None, None


    fig = plt.figure(figsize=figsize)


    ax1 = fig.add_subplot(221, projection='3d')

    epochs = filtered_data['epoch'].values
    loss_values = filtered_data['total_loss'].values

    ax1.plot(epochs, np.zeros_like(epochs), loss_values, 'b-', linewidth=3, alpha=0.8, label='Total Loss')

    scatter1 = ax1.scatter(epochs, np.zeros_like(epochs), loss_values,
                           c=loss_values, cmap='viridis', s=30, alpha=0.7)

    ax1.set_xlabel('Epoch', labelpad=10)
    ax1.set_ylabel('Y', labelpad=10)
    ax1.set_zlabel('Total Loss', labelpad=10)
    ax1.set_title('3D Total Loss Curve', fontsize=12, pad=20)
    ax1.legend()

    plt.colorbar(scatter1, ax=ax1, shrink=0.6, aspect=10, label='Loss Value')

    ax2 = fig.add_subplot(222, projection='3d')

    dx = 0.8
    dy = 0.1
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))

    for i, (epoch, loss) in enumerate(zip(epochs, loss_values)):
        ax2.bar3d(epoch, 0, 0, dx, dy, loss,
                  color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.3)

    ax2.set_xlabel('Epoch', labelpad=10)
    ax2.set_ylabel('Y', labelpad=10)
    ax2.set_zlabel('Total Loss', labelpad=10)
    ax2.set_title('3D Loss Bars', fontsize=12, pad=20)


    ax3 = fig.add_subplot(223, projection='3d')


    if len(epochs) > 1:

        X = np.array([epochs, epochs]).T
        Y = np.array([np.zeros_like(epochs), np.ones_like(epochs) * 0.5]).T
        Z = np.array([loss_values, loss_values]).T


        surf = ax3.plot_surface(X, Y, Z, cmap='plasma', alpha=0.6, edgecolor='none')

        ax3.set_xlabel('Epoch', labelpad=10)
        ax3.set_ylabel('Y', labelpad=10)
        ax3.set_zlabel('Total Loss', labelpad=10)
        ax3.set_title('3D Loss Surface', fontsize=12, pad=20)


        plt.colorbar(surf, ax=ax3, shrink=0.6, aspect=10, label='Loss Value')


    ax4 = fig.add_subplot(224, projection='3d')

    loss_diff = np.diff(loss_values, prepend=loss_values[0])
    colors_diff = plt.cm.RdYlGn((loss_diff - loss_diff.min()) / (loss_diff.max() - loss_diff.min() + 1e-8))

    scatter4 = ax4.scatter(epochs, loss_diff, loss_values,
                           c=loss_values, cmap='coolwarm', s=50, alpha=0.8)


    ax4.plot(epochs, loss_diff, loss_values, 'k-', alpha=0.3, linewidth=0.5)

    ax4.set_xlabel('Epoch', labelpad=10)
    ax4.set_ylabel('Loss Change', labelpad=10)
    ax4.set_zlabel('Total Loss', labelpad=10)
    ax4.set_title('3D Loss with Change Rate', fontsize=12, pad=20)


    plt.colorbar(scatter4, ax=ax4, shrink=0.6, aspect=10, label='Loss Value')

    plt.suptitle('YOLO Total Loss 3D Visualization (Combined Loss)',
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()


    if save_path:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{save_path}_3d_total_loss_simple.png", dpi=300, bbox_inches='tight')
        print(f"3D总损失图已保存到: {save_path}_3d_total_loss_simple.png")

    plt.show()

    return fig, filtered_data


def plot_total_loss_3d_alternative(csv_path, save_path=None, start_epoch=0, end_epoch=None, figsize=(14, 10)):

    total_loss_df, loss_components = calculate_total_loss(csv_path)
    if total_loss_df is None:
        return None

    if end_epoch is None:
        end_epoch = len(total_loss_df) - 1

    mask = (total_loss_df['epoch'] >= start_epoch) & (total_loss_df['epoch'] <= end_epoch)
    filtered_data = total_loss_df[mask].copy()

    if len(filtered_data) < 3:
        print("错误: 数据点太少")
        return None

    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(121, projection='3d')

    epochs = filtered_data['epoch'].values
    loss_values = filtered_data['total_loss'].values


    theta = np.linspace(0, 4 * np.pi, len(epochs))
    radius = 1.0

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z_bottom = np.zeros_like(epochs)

    dx = dy = 0.1
    max_loss = max(loss_values) if max(loss_values) > 0 else 1

    for i, (xi, yi, loss) in enumerate(zip(x, y, loss_values)):
        color = plt.cm.viridis(loss / max_loss)
        ax1.bar3d(xi, yi, z_bottom[i], dx, dy, loss,
                  color=color, alpha=0.8, edgecolor='black', linewidth=0.3)

    ax1.set_xlabel('X', labelpad=10)
    ax1.set_ylabel('Y', labelpad=10)
    ax1.set_zlabel('Total Loss', labelpad=10)
    ax1.set_title('3D Spiral Loss Bars', fontsize=12, pad=20)


    ax2 = fig.add_subplot(122, projection='3d')


    angles = np.linspace(0, 2 * np.pi, 36)
    funnel_x = []
    funnel_y = []
    funnel_z = []

    for i, epoch in enumerate(epochs):

        if max_loss > 0:
            radius = 0.5 * (1 - loss_values[i] / max_loss) + 0.1
        else:
            radius = 0.5


        circle_x = radius * np.cos(angles)
        circle_y = radius * np.sin(angles)
        circle_z = np.full_like(angles, epoch)

        funnel_x.append(circle_x)
        funnel_y.append(circle_y)
        funnel_z.append(circle_z)


    funnel_x = np.array(funnel_x)
    funnel_y = np.array(funnel_y)
    funnel_z = np.array(funnel_z)


    for i in range(len(epochs) - 1):
        for j in range(len(angles) - 1):

            vertices = [
                [funnel_x[i, j], funnel_y[i, j], funnel_z[i, j]],
                [funnel_x[i, j + 1], funnel_y[i, j + 1], funnel_z[i, j + 1]],
                [funnel_x[i + 1, j + 1], funnel_y[i + 1, j + 1], funnel_z[i + 1, j + 1]],
                [funnel_x[i + 1, j], funnel_y[i + 1, j], funnel_z[i + 1, j]]
            ]


            avg_loss = (loss_values[i] + loss_values[i + 1]) / 2
            color = plt.cm.plasma(avg_loss / max_loss)


            from matplotlib.patches import Polygon
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            poly = Poly3DCollection([vertices], alpha=0.3)
            poly.set_facecolor(color)
            poly.set_edgecolor('black')
            poly.set_linewidth(0.1)
            ax2.add_collection3d(poly)


    ax2.plot(np.zeros_like(epochs), np.zeros_like(epochs), epochs,
             'r-', linewidth=2, label='Center Line')

    ax2.set_xlabel('X', labelpad=10)
    ax2.set_ylabel('Y', labelpad=10)
    ax2.set_zlabel('Epoch', labelpad=10)
    ax2.set_title('3D Loss Funnel', fontsize=12, pad=20)
    ax2.legend()

    plt.suptitle('Alternative 3D Visualizations of Total Loss',
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()


    if save_path:
        plt.savefig(f"{save_path}_3d_alternative.png", dpi=300, bbox_inches='tight')
        print(f"替代3D图已保存到: {save_path}_3d_alternative.png")

    plt.show()

    return fig


def plot_total_loss_2d_with_3d_effect(csv_path, save_path=None, start_epoch=0, end_epoch=None, figsize=(16, 12)):

    total_loss_df, loss_components = calculate_total_loss(csv_path)
    if total_loss_df is None:
        return None


    if end_epoch is None:
        end_epoch = len(total_loss_df) - 1

    mask = (total_loss_df['epoch'] >= start_epoch) & (total_loss_df['epoch'] <= end_epoch)
    filtered_data = total_loss_df[mask].copy()


    fig, axes = plt.subplots(2, 3, figsize=figsize)

    epochs = filtered_data['epoch'].values
    loss_values = filtered_data['total_loss'].values
    loss_smooth = filtered_data['loss_smooth'].values
    loss_change = filtered_data['loss_change'].values


    ax1 = axes[0, 0]
    ax1.plot(epochs, loss_values, 'b-', linewidth=2, label='Total Loss')
    ax1.plot(epochs, loss_smooth, 'r--', linewidth=1.5, alpha=0.7, label='Smoothed')
    ax1.fill_between(epochs, 0, loss_values, alpha=0.2, color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss Curve')
    ax1.grid(True, alpha=0.3)
    ax1.legend()


    ax2 = axes[0, 1]
    colors = ['green' if x < 0 else 'red' for x in loss_change]
    ax2.bar(epochs, loss_change, color=colors, alpha=0.6, width=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Change')
    ax2.set_title('Loss Change per Epoch')
    ax2.grid(True, alpha=0.3)


    ax3 = axes[0, 2]
    scatter = ax3.scatter(epochs, loss_values, c=loss_values, cmap='viridis',
                          s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax3.plot(epochs, loss_values, 'b-', alpha=0.3)

    for i in range(len(epochs) - 1):
        ax3.fill_between([epochs[i], epochs[i + 1]],
                         [loss_values[i], loss_values[i + 1]],
                         0, alpha=0.1, color='blue')

    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Total Loss')
    ax3.set_title('Loss with 3D-like Depth Effect')
    ax3.grid(True, alpha=0.3)


    ax4 = axes[1, 0]


    if len(epochs) > 10:
        x = epochs
        y = np.linspace(0, max(loss_values) * 0.2, 10)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)


        for i in range(len(x)):
            for j in range(len(y)):

                Z[j, i] = loss_values[i] * np.exp(-(y[j] ** 2) / (2 * (max(loss_values) * 0.05) ** 2))


        contour = ax4.contourf(X, Y, Z, levels=20, cmap='plasma', alpha=0.8)
        ax4.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5, alpha=0.5)


        ax4.plot(epochs, np.zeros_like(epochs), 'w-', linewidth=2, alpha=0.7)

    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Height')
    ax4.set_title('Loss Terrain Map')


    ax5 = axes[1, 1]


    if len(loss_components) > 0:

        bottom = np.zeros_like(epochs)

        for i, col in enumerate(loss_components[:3]):
            if col in total_loss_df.columns:
                data = pd.to_numeric(total_loss_df[col], errors='coerce').fillna(0).values
                data = data[mask]
                ax5.fill_between(epochs, bottom, bottom + data,
                                 alpha=0.5, label=f'Loss {i + 1}')
                bottom += data

        ax5.plot(epochs, loss_values, 'k-', linewidth=1.5, label='Total Loss')
    else:

        ax5.plot(epochs, loss_values, 'b-', linewidth=2, label='Total Loss')
        ax5.fill_between(epochs, 0, loss_values, alpha=0.3, color='blue')

    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss Value')
    ax5.set_title('Loss Components (if available)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)


    ax6 = axes[1, 2]
    ax6.axis('off')


    final_loss = loss_values[-1] if len(loss_values) > 0 else 0
    min_loss = min(loss_values) if len(loss_values) > 0 else 0
    max_loss = max(loss_values) if len(loss_values) > 0 else 0
    avg_loss = np.mean(loss_values) if len(loss_values) > 0 else 0

    if len(loss_values) > 1 and loss_values[0] > 0:
        loss_reduction = ((loss_values[0] - final_loss) / loss_values[0] * 100)
    else:
        loss_reduction = 0

    stats_text = f"""
    YOLO Total Loss Analysis

    Data Source: {Path(csv_path).name}
    Total Epochs: {len(filtered_data)}
    Loss Components: {len(loss_components)}

    Loss Statistics:
    - Final Loss: {final_loss:.4f}
    - Minimum Loss: {min_loss:.4f}
    - Maximum Loss: {max_loss:.4f}
    - Average Loss: {avg_loss:.4f}

    Training Progress:
    - Loss Reduction: {loss_reduction:.1f}%
    - Avg Change per Epoch: {np.mean(loss_change):.4f}
    """

    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('YOLO Total Loss Analysis with 3D Effects',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()


    if save_path:
        plt.savefig(f"{save_path}_2d_3d_effect.png", dpi=300, bbox_inches='tight')
        print(f"2D带3D效果图已保存到: {save_path}_2d_3d_effect.png")

    plt.show()

    return fig



if __name__ == "__main__":

    csv_file_path = "resualts.csv"

    if not Path(csv_file_path).exists():
        print(f"错误: 文件不存在 - {csv_file_path}")
        print("请确保文件路径正确，或提供完整路径")
        print("示例: C:/Users/用户名/Desktop/yolo_results/results.csv")
    else:
        try:
            print("=" * 60)
            print("开始绘制YOLO总损失可视化图")
            print("=" * 60)


            print("\n1. 绘制简单3D总损失图...")
            fig1, data1 = plot_total_loss_3d_simple(
                csv_path=csv_file_path,
                save_path="yolo_total_loss",
                start_epoch=0,
                end_epoch=None,
                figsize=(14, 10)
            )


            print("\n2. 绘制替代3D图...")
            fig2 = plot_total_loss_3d_alternative(
                csv_path=csv_file_path,
                save_path="yolo_total_loss",
                start_epoch=0,
                end_epoch=100,
                figsize=(14, 10)
            )


            print("\n3. 绘制2D带3D效果图...")
            fig3 = plot_total_loss_2d_with_3d_effect(
                csv_path=csv_file_path,
                save_path="yolo_total_loss",
                start_epoch=0,
                end_epoch=None,
                figsize=(16, 12)
            )

            print("\n" + "=" * 60)
            print("所有图形绘制完成!")
            print("=" * 60)


            if data1 is not None:
                data1.to_csv("total_loss_data.csv", index=False, encoding='utf-8-sig')
                print("总损失数据已保存到: total_loss_data.csv")

        except Exception as e:
            print(f"\n绘图时出错: {e}")
            import traceback

            traceback.print_exc()

            print("\n" + "=" * 60)
            print("简化版本，直接绘制2D图:")
            print("=" * 60)


            try:
                total_loss_df, loss_components = calculate_total_loss(csv_file_path)
                if total_loss_df is not None:
                    plt.figure(figsize=(12, 6))
                    plt.plot(total_loss_df['epoch'], total_loss_df['total_loss'], 'b-', linewidth=2)
                    plt.xlabel('Epoch')
                    plt.ylabel('Total Loss')
                    plt.title(f'YOLO Total Loss (Combined {len(loss_components)} components)')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig("yolo_total_loss_simple.png", dpi=300, bbox_inches='tight')
                    plt.show()
                    print("简单2D图已保存到: yolo_total_loss_simple.png")
            except:
                print("无法绘制任何图形，请检查CSV文件格式。")