import web3 from './web3';
import { jobs} from "../abi/abi";

const address = '0x12F0F455D3e769b247518747dd731E9c61366E97';

export default new web3.eth.Contract(jobs, address);
