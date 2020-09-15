def ilcm_attack(img, ori_label, model,  eps, step_size, max_epoch):
    img_min = clip_by_tensor(img - 2.0 * eps, -1.0, 1.0)
    img_max = clip_by_tensor(img + 2.0 * eps, -1.0, 1.0)
    gt_output = model(img)
    target_label = gt_output.argmin(1).to(device)
    print('The target label is: {}'.format(target_label.item()))
    image = img.clone()
    image.requires_grad = True
    for i in range(max_epoch):
        zero_gradients(image)
        output = model(image)
        pert_label = output.argmax(1)
        if pert_label == target_label:
            break
        loss = F.cross_entropy(output, target_label)
        loss.backward()
        grad = image.grad.data
        image = image.data - step_size * torch.sign(grad)
        image = clip_by_image(image.data, img_min, img_max)
        image = V(image, requires_grad=True)
    return image.detach(), pert_label, i + 1, target_label
